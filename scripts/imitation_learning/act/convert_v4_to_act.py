"""Convert v4 per-demo HD hdf5s to ACT per-episode format with image resize.

v4: /mnt/Dataset/robotis_outputs/callbutton_demos_50_hd_v4/demo_NN.hdf5
    - root-level /actions, /obs/{cam_wrist,cam_top,cam_belly, joint_pos, joint_vel}
    - images HD (1200x1920)
ACT: episode_N.hdf5 with /action, /observations/{qpos, qvel, images/<cam>}
    - images resized to target_hw
"""
import os
import glob
import argparse
import h5py
import numpy as np
from PIL import Image


def resize_frames(arr, target_hw):
    """arr: (T, H, W, 3) uint8 -> (T, target_h, target_w, 3) uint8."""
    T = arr.shape[0]
    h, w = target_hw
    out = np.empty((T, h, w, 3), dtype=np.uint8)
    for i in range(T):
        out[i] = np.asarray(Image.fromarray(arr[i]).resize((w, h), Image.BILINEAR))
    return out


def convert(src_dir, out_dir, cam_names, target_hw, min_size_bytes=1_000_000, episode_len=None, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "demo_*.hdf5")))
    valid = [f for f in files if os.path.getsize(f) > min_size_bytes]
    if limit is not None:
        valid = valid[:limit]
    print(f"found {len(files)} files, using {len(valid)} (limit={limit})")

    # First pass: length scan
    lengths = []
    for f in valid:
        with h5py.File(f, "r") as h:
            demo = list(h["data"].keys())[0]
            lengths.append(h[f"data/{demo}/actions"].shape[0])
    if episode_len is None:
        episode_len = min(lengths)
    print(f"lengths min/max/mean: {min(lengths)}/{max(lengths)}/{sum(lengths)//len(lengths)} -> unified={episode_len}")

    for i, f in enumerate(valid):
        with h5py.File(f, "r") as h:
            demo = list(h["data"].keys())[0]
            d = h[f"data/{demo}"]
            T = d["actions"].shape[0]
            sl = slice(0, min(T, episode_len))
            pad_n = episode_len - min(T, episode_len)

            def pad(arr):
                if pad_n == 0:
                    return arr
                tail = np.tile(arr[-1:], (pad_n,) + (1,) * (arr.ndim - 1))
                return np.concatenate([arr, tail], axis=0)

            out_path = os.path.join(out_dir, f"episode_{i}.hdf5")
            with h5py.File(out_path, "w") as out:
                out.attrs["sim"] = True
                obs = out.create_group("observations")
                obs.create_dataset("qpos", data=pad(d["obs/joint_pos"][sl][:, :7]))
                obs.create_dataset("qvel", data=pad(d["obs/joint_vel"][sl][:, :7]))
                imgs = obs.create_group("images")
                for cam in cam_names:
                    raw = d[f"obs/{cam}"][sl]  # (T, H, W, 3)
                    resized = resize_frames(raw, target_hw)
                    imgs.create_dataset(cam, data=pad(resized), compression="gzip", compression_opts=1)
                out.create_dataset("action", data=pad(d["actions"][sl]))
        print(f"  [{i+1}/{len(valid)}] {os.path.basename(f)} T={T} -> episode_{i}.hdf5", flush=True)

    print(f"done. {len(valid)} episodes (len={episode_len}) -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cams", default="cam_wrist,cam_top,cam_belly")
    p.add_argument("--h", type=int, default=240)
    p.add_argument("--w", type=int, default=320)
    p.add_argument("--episode_len", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    convert(a.src_dir, a.out, a.cams.split(","), (a.h, a.w), episode_len=a.episode_len, limit=a.limit)
