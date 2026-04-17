#!/bin/bash
# Verify demo HDF5 structure + extract sample frames.
set -e
HDF5=${1:-/isaac-sim/output/callbutton_demos_cam/demo_00_usd001.hdf5}
OUT=${OUT:-/isaac-sim/output/demo_verify}

mkdir -p "${OUT}"

/isaac-sim/python.sh - <<PYEOF
import h5py, os, numpy as np
from PIL import Image

path = "${HDF5}"
out = "${OUT}"
f = h5py.File(path, "r")
demo = f["data"][list(f["data"].keys())[0]]

print("=" * 60)
print(f"FILE: {path}")
print(f"File size: {os.path.getsize(path) / 1e6:.1f} MB")
print(f"Demos: {list(f['data'].keys())}")
print(f"demo_0 top-level keys: {list(demo.keys())}")
print()

print("=== obs ===")
for k in demo["obs"].keys():
    v = demo["obs"][k]
    if hasattr(v, "shape"):
        print(f"  obs[{k}]: {v.shape}  {v.dtype}")

print()
print("=== top-level fields ===")
for k in demo.keys():
    v = demo[k]
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape}  {v.dtype}")
    else:
        print(f"  {k}: group")

# Initial state structure
if "initial_state" in demo:
    print()
    print("=== initial_state ===")
    def visit(name, obj):
        if hasattr(obj, "shape"):
            print(f"  initial_state/{name}: {obj.shape}  {obj.dtype}")
    demo["initial_state"].visititems(visit)

# Dump preview frames
print()
print("=== saving preview frames ===")
for cam in ["cam_top", "cam_wrist"]:
    if cam not in demo["obs"]:
        continue
    arr = demo["obs"][cam]
    T = arr.shape[0]
    for i, t in enumerate([0, T // 4, T // 2, 3 * T // 4, T - 1]):
        img = np.array(arr[t])
        fn = f"{out}/{cam}_t{i}_{t:04d}.png"
        Image.fromarray(img).save(fn)
        print(f"  {fn}")

print()
print("=" * 60)
PYEOF
