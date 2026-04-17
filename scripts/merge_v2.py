# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

"""Proper external-link merge — creates tiny 4KB virtual HDF5."""
import glob, h5py, os

SRC_DIR = "/isaac-sim/output/callbutton_demos_cam"
DST = f"{SRC_DIR}/merged.hdf5"

if os.path.exists(DST):
    os.remove(DST)

files = sorted(glob.glob(f"{SRC_DIR}/demo_*_usd*.hdf5"))
print(f"Merging {len(files)} files via ExternalLink")

with h5py.File(DST, "w") as dst:
    data_grp = dst.create_group("data")
    # Propagate env_args attribute from first file (robomimic needs it).
    with h5py.File(files[0], "r") as src:
        if "data" in src and "env_args" in src["data"].attrs:
            data_grp.attrs["env_args"] = src["data"].attrs["env_args"]
        elif "data" in src:
            for k, v in src["data"].attrs.items():
                data_grp.attrs[k] = v
    for i, f in enumerate(files):
        dst[f"data/demo_{i}"] = h5py.ExternalLink(f, "data/demo_0")
        print(f"  demo_{i} → {os.path.basename(f)}")

with h5py.File(DST, "r") as f:
    print(f"\nSize: {os.path.getsize(DST)/1024:.1f} KB  demos: {list(f['data'].keys())}")
    d0 = f["data/demo_0"]
    print(f"demo_0/actions: {d0['actions'].shape}  obs keys: {list(d0['obs'].keys())}")
