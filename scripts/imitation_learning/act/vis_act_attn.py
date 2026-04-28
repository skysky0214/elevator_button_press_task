"""Visualize ACT decoder cross-attention as heatmap overlay on camera frames.

Captures the last transformer-decoder layer's cross-attention weights, which show
where each of the 50 action queries is looking in the 3-camera feature map.
Averages heads and queries, reshapes to per-camera (H, W) grids, upsamples, and
overlays as a colored heatmap on the original 240x320 image.

Writes an MP4 of the full demo (frames stacked: wrist | top | belly).
"""
import argparse
import os
import sys
import pickle
import subprocess

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CAMERA_NAMES = ["cam_wrist", "cam_top", "cam_belly"]


def colorize(x, cmap_name="inferno"):
    """x: (H, W) in [0, 1] → (H, W, 3) uint8."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(x.clip(0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(img_u8, heat_u8, alpha=0.55):
    return (img_u8.astype(np.float32) * (1 - alpha) + heat_u8.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--dataset", required=True, help="ACT-format episode_N.hdf5")
    p.add_argument("--act_repo", default="/mnt/Dataset/act_repo")
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img_h", type=int, default=240)
    p.add_argument("--img_w", type=int, default=320)
    p.add_argument("--out", default="/tmp/act_attn.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    args = p.parse_args()

    sys.path.insert(0, args.act_repo)
    sys.path.insert(0, os.path.join(args.act_repo, "detr"))

    # Replace sys.argv entirely so DETR's build_parser inside policy doesn't choke
    sys.argv = [sys.argv[0],
                '--ckpt_dir', '/tmp/_dp_dummy',
                '--policy_class', 'ACT',
                '--task_name', 'robotis_callbutton',
                '--seed', '0',
                '--num_epochs', '1']

    from policy import ACTPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy({
        "lr": 1e-5, "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight, "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5, "backbone": args.backbone,
        "enc_layers": 4, "dec_layers": 7, "nheads": 8,
        "camera_names": CAMERA_NAMES,
    })
    sd = torch.load(args.ckpt, map_location=device)
    policy.load_state_dict(sd)
    policy.to(device).eval()

    # Hook the LAST decoder layer's cross-attention (multihead_attn) to capture weights.
    decoder_layers = policy.model.transformer.decoder.layers
    last_layer = decoder_layers[-1]
    captured = {}

    def attn_hook(module, inp, out):
        # out = (attn_output, attn_weights)
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            captured["w"] = out[1].detach()  # (B, num_queries, src_len) or with heads

    last_layer.multihead_attn.register_forward_hook(attn_hook)

    # Force need_weights=True (otherwise weights returned as None in some torch versions)
    orig_fwd = last_layer.multihead_attn.forward
    def new_fwd(*a, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = True
        return orig_fwd(*a, **kw)
    last_layer.multihead_attn.forward = new_fwd

    # Normalization stats
    with open(args.stats, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
    qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)

    # Load demo
    h = h5py.File(args.dataset, "r")
    T = h["/action"].shape[0]
    if args.max_frames and args.max_frames < T:
        T = args.max_frames
    print(f"demo T={T}, camera grid to be inferred from first frame", flush=True)

    frames_dir = os.path.splitext(args.out)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    Hf = Wf = None
    for t in range(T):
        qpos_raw = h["/observations/qpos"][t]  # (8,)
        qpos = torch.from_numpy(qpos_raw).float().unsqueeze(0).to(device)
        qpos_norm = (qpos - qpos_mean) / qpos_std

        imgs_u8 = [h[f"/observations/images/{cam}"][t] for cam in CAMERA_NAMES]
        imgs_t = []
        for im in imgs_u8:
            x = torch.from_numpy(im).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
            if x.shape[-2] != args.img_h or x.shape[-1] != args.img_w:
                x = F.interpolate(x, size=(args.img_h, args.img_w), mode="bilinear", align_corners=False)
            imgs_t.append(x)
        image = torch.stack(imgs_t, dim=1)  # (1, 3cam, 3C, H, W)

        captured.clear()
        with torch.no_grad():
            _ = policy(qpos_norm, image)

        if "w" not in captured:
            raise RuntimeError("attention not captured; hook failed")
        # w: (B, num_queries, src_len)
        w = captured["w"]
        # average over queries → (B, src_len)
        w_avg = w.mean(dim=1).squeeze(0)  # (src_len,)

        # src layout = [latent(1), proprio(1), img_tokens(H*W*3cam)]
        src_len = w_avg.shape[0]
        img_len = src_len - 2
        if Hf is None:
            # infer per-cam grid from backbone: with ResNet18 stride 32 on 240x320 → 8x10 = 80
            # 3 cams → 240 tokens. But we don't know Hf/Wf a priori. Infer from math:
            # We know image resolution and backbone stride.
            # ACT concats cams along WIDTH, so total spatial = Hf x (Wf*3).
            per_cam = img_len // 3
            # Try factors near input/32
            # for resnet18 stride 32: Hf = ceil(H/32), Wf = ceil(W/32) roughly
            hh = args.img_h // 32
            ww = args.img_w // 32
            if hh * ww * 3 != img_len:
                # try 1 more row (e.g. 8x10 on 240x320)
                for hh in range(max(1, args.img_h // 32 - 1), args.img_h // 32 + 2):
                    ww = per_cam // hh
                    if hh * ww == per_cam:
                        break
            Hf, Wf = hh, ww
            print(f"inferred feature grid: Hf={Hf}, Wf={Wf}, per_cam={per_cam}, total_img={img_len}", flush=True)

        img_attn = w_avg[2:].reshape(Hf, Wf * 3).cpu().numpy()  # (Hf, Wf*3) concat along width
        # per-cam slices
        panels = []
        for ci, cam in enumerate(CAMERA_NAMES):
            slab = img_attn[:, ci * Wf:(ci + 1) * Wf]
            slab_t = torch.from_numpy(slab).float().unsqueeze(0).unsqueeze(0)
            up = F.interpolate(slab_t, size=(args.img_h, args.img_w), mode="bilinear", align_corners=False)[0, 0].numpy()
            # normalize per-cam
            up = (up - up.min()) / (up.max() - up.min() + 1e-6)
            heat = colorize(up)
            # original image (resize to img_h x img_w)
            orig = imgs_u8[ci]
            orig_pil = Image.fromarray(orig).resize((args.img_w, args.img_h), Image.BILINEAR)
            overlayed = overlay(np.asarray(orig_pil), heat, alpha=0.55)
            panels.append(overlayed)

        # stack panels horizontally (wrist | top | belly)
        strip = np.concatenate(panels, axis=1)  # (H, W*3, 3)
        # Add label
        Image.fromarray(strip).save(os.path.join(frames_dir, f"f_{t:05d}.png"))
        if t % 50 == 0:
            print(f"  {t}/{T}", flush=True)

    h.close()

    # ffmpeg stitch
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(frames_dir, "f_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg stderr:", r.stderr[-2000:])
        raise SystemExit(r.returncode)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
