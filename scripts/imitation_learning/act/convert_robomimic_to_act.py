"""Convert robomimic-format merged hdf5 to ACT per-episode hdf5s.

robomimic: /data/demo_N/{actions, obs/{joint_pos, joint_vel, cam_*, ...}}
ACT:       episode_N.hdf5 with /action, /observations/{qpos, qvel, images/<cam>}
"""
import os
import argparse
import h5py
import numpy as np


def convert(src_path, out_dir, cam_names, episode_len=None):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(src_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        # Determine common length
        lengths = [f[f"data/{k}/actions"].shape[0] for k in demo_keys]
        if episode_len is None:
            episode_len = min(lengths)
        print(f"lengths min/max/mean: {min(lengths)}/{max(lengths)}/{sum(lengths)//len(lengths)}  -> unified={episode_len}")
        for i, k in enumerate(demo_keys):
            d = f[f"data/{k}"]
            T = d["actions"].shape[0]
            if T >= episode_len:
                sl = slice(0, episode_len)
                pad_needed = 0
            else:
                sl = slice(0, T)
                pad_needed = episode_len - T
            out_path = os.path.join(out_dir, f"episode_{i}.hdf5")
            with h5py.File(out_path, "w") as out:
                out.attrs["sim"] = True
                obs_grp = out.create_group("observations")

                def pad_last(arr):
                    if pad_needed == 0:
                        return arr
                    tail = np.tile(arr[-1:], (pad_needed,) + (1,) * (arr.ndim - 1))
                    return np.concatenate([arr, tail], axis=0)

                # ACT expects qpos_dim == action_dim (state_dim) — slice 10-dim joint_pos to 7
                obs_grp.create_dataset("qpos", data=pad_last(d["obs/joint_pos"][sl][:, :7]))
                obs_grp.create_dataset("qvel", data=pad_last(d["obs/joint_vel"][sl][:, :7]))
                img_grp = obs_grp.create_group("images")
                for cam in cam_names:
                    img_grp.create_dataset(cam, data=pad_last(d[f"obs/{cam}"][sl]))
                out.create_dataset("action", data=pad_last(d["actions"][sl]))
            print(f"  episode_{i}.hdf5  orig_T={T}  pad={pad_needed}  action_dim={d['actions'].shape[1]}  qpos_dim={d['obs/joint_pos'].shape[1]}")
    print(f"converted {len(demo_keys)} episodes (len={episode_len}) -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cams", default="cam_wrist,cam_top,cam_belly")
    p.add_argument("--episode_len", type=int, default=None, help="If None, uses min across demos")
    a = p.parse_args()
    convert(a.src, a.out, a.cams.split(","), episode_len=a.episode_len)
