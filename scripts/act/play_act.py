"""Run a trained ACT policy in Isaac Lab (callbutton task).

Loads:
    - ACT checkpoint (state_dict)
    - dataset_stats.pkl (qpos/action mean & std)
    - callbutton env (Isaac Sim)

Per-step flow:
    qpos = joint_pos[:7] (normalized)
    image = cam_wrist/top/belly (resized to match training, ImageNet-normalized inside policy)
    a_hat = policy(qpos, image)   # (1, chunk_size, 7), normalized
    a_exec = chunk[0] (or temporal ensemble)  -> un-normalize -> env.step
"""

import argparse
import os
import sys
import pickle

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACT policy on Isaac Lab.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True, help="Path to policy_*.ckpt")
parser.add_argument("--stats", type=str, required=True, help="Path to dataset_stats.pkl")
parser.add_argument("--act_repo", type=str, default="/mnt/Dataset/act_repo")
parser.add_argument("--chunk_size", type=int, default=50)
parser.add_argument("--kl_weight", type=int, default=10)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--dim_feedforward", type=int, default=3200)
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--num_rollouts", type=int, default=5)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--temporal_agg", action="store_true", help="Use temporal ensembling over overlapping chunks")
parser.add_argument("--img_h", type=int, default=240)
parser.add_argument("--img_w", type=int, default=320)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Replace sys.argv entirely before DETR's parser runs — only the DETR-required flags
import sys as _sys
_sys.argv = [_sys.argv[0],
             '--ckpt_dir', '/tmp/_dp_dummy',
             '--policy_class', 'ACT',
             '--task_name', 'robotis_callbutton',
             '--seed', '0',
             '--num_epochs', '1']

"""Rest of imports."""

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

# Make ACT repo importable
sys.path.insert(0, args_cli.act_repo)
sys.path.insert(0, os.path.join(args_cli.act_repo, "detr"))
from policy import ACTPolicy  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import robotis_lab  # noqa: F401,E402


CAMERA_NAMES = ["cam_wrist", "cam_top", "cam_belly"]


def build_policy(device):
    policy_config = {
        "lr": 1e-5,
        "num_queries": args_cli.chunk_size,
        "kl_weight": args_cli.kl_weight,
        "hidden_dim": args_cli.hidden_dim,
        "dim_feedforward": args_cli.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": CAMERA_NAMES,
    }
    policy = ACTPolicy(policy_config)
    sd = torch.load(args_cli.ckpt, map_location=device)
    policy.load_state_dict(sd)
    policy.to(device).eval()
    return policy


def prep_obs(obs_dict, qpos_mean, qpos_std, device):
    """obs_dict is env.observations.policy dict. Return (qpos_norm, image_tensor)."""
    policy_obs = obs_dict["policy"]
    joint_pos = policy_obs["joint_pos"]  # (B, 10)
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(0)
    jp7 = joint_pos[:, :7].float()
    # LED obs (added in v5): concat as 8th qpos channel
    led = policy_obs.get("call_button_lit")
    if led is None:
        led = torch.zeros((jp7.shape[0], 1), device=jp7.device)
    else:
        if led.ndim == 1:
            led = led.unsqueeze(0)
        led = led.float().reshape(jp7.shape[0], 1)
    qpos = torch.cat([jp7, led], dim=1)
    qpos = (qpos - qpos_mean) / qpos_std

    imgs = []
    for cam in CAMERA_NAMES:
        im = policy_obs[cam]  # (B,H,W,3) uint8 or float
        if im.ndim == 3:
            im = im.unsqueeze(0)
        im = im.float()
        # to (B,3,H,W)
        im = im.permute(0, 3, 1, 2).contiguous()
        if im.max() > 1.5:
            im = im / 255.0
        # resize to (img_h, img_w) to match training
        if im.shape[-2] != args_cli.img_h or im.shape[-1] != args_cli.img_w:
            im = F.interpolate(im, size=(args_cli.img_h, args_cli.img_w), mode="bilinear", align_corners=False)
        imgs.append(im)
    # stack (B, num_cams, 3, H, W)
    image = torch.stack(imgs, dim=1).to(device)
    qpos = qpos.to(device)
    return qpos, image


def rollout(policy, env, success_term, horizon, device, stats):
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
    qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)
    action_mean = torch.from_numpy(stats["action_mean"]).float().to(device)
    action_std = torch.from_numpy(stats["action_std"]).float().to(device)

    policy.eval()
    obs_dict, _ = env.reset()

    chunk_size = args_cli.chunk_size
    action_dim = action_mean.shape[0]

    # Temporal ensemble buffer: at time t, store predicted actions for [t, t+chunk_size)
    if args_cli.temporal_agg:
        all_time_actions = torch.zeros(horizon, horizon + chunk_size, action_dim, device=device)
    pending = []  # plain chunk buffer when not temporal_agg

    for t in range(horizon):
        qpos, image = prep_obs(obs_dict, qpos_mean, qpos_std, device)

        need_predict = args_cli.temporal_agg or len(pending) == 0
        if need_predict:
            with torch.no_grad():
                a_hat = policy(qpos, image)  # (1, chunk_size, 7) normalized
            if args_cli.temporal_agg:
                # write into buffer at rows [t], cols [t:t+chunk_size]
                end = min(t + chunk_size, all_time_actions.shape[1])
                all_time_actions[t, t:end] = a_hat[0, : end - t]
            else:
                pending = [a_hat[0, k] for k in range(chunk_size)]

        if args_cli.temporal_agg:
            actions_for_t = all_time_actions[: t + 1, t]  # (<=t+1, action_dim)
            actions_populated = torch.any(actions_for_t != 0, dim=1)
            actions_for_t = actions_for_t[actions_populated]
            k = 0.01
            exp_w = torch.exp(-k * torch.arange(actions_for_t.shape[0], device=device).float())
            exp_w = exp_w / exp_w.sum()
            a = (actions_for_t * exp_w.unsqueeze(1)).sum(dim=0)
        else:
            a = pending.pop(0)

        # un-normalize
        a_raw = a * action_std + action_mean  # (8,) — 8th is dummy

        if t % 40 == 0:
            print(f"[ACT] t={t} a_raw={a_raw.cpu().numpy().round(4).tolist()}", flush=True)

        # env expects 7-dim action; slice off dummy 8th channel
        action_cmd = a_raw[:7].unsqueeze(0).to(device=device, dtype=torch.float32)
        obs_dict, _, terminated, truncated, _ = env.step(action_cmd)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, t
        if terminated or truncated:
            return False, t
    return False, horizon


def main():
    with open(args_cli.stats, "rb") as f:
        stats = pickle.load(f)
    print("[ACT] stats loaded:",
          {k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in stats.items()}, flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(device)
    print("[ACT] policy loaded. state_dim=7, chunk_size=", args_cli.chunk_size, flush=True)

    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[ACT] Trial {trial}", flush=True)
        ok, steps = rollout(policy, env, success_term, args_cli.horizon, device, stats)
        results.append(ok)
        print(f"[ACT] Trial {trial}: {ok} (steps={steps})", flush=True)

    print(f"Success: {results.count(True)}/{len(results)} = {results.count(True)/len(results):.1%}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
