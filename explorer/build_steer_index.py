"""Stacks the per-layer steer lens files from sweep/steer_lens.py into the explorer's steer index.

    uv run python explorer/build_steer_index.py --stats stats/steer_lens

Reads stats/steer_lens_L{l}.npz for every layer (all must exist) and writes explorer/cache/steer_index.npz:
up[L, N, k], down[L, N, k] (int32 token ids), up_kl[L, N], down_kl[L, N] (float16), N the widest layer (narrower
layers are padded with token 0 and KL 0).
"""
import argparse
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stats", default="stats/steer_lens", help="prefix given to steer_lens.py --out")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "steer_index.npz"))
a = ap.parse_args()

paths = {int(re.search(r"_L(\d+)\.npz$", p).group(1)): p for p in glob.glob(f"{a.stats}_L*.npz")}
L = max(paths) + 1
missing = sorted(set(range(L)) - set(paths))
assert not missing, f"steer lens missing for layers {missing}"
z = [np.load(paths[l]) for l in range(L)]
N = max(len(x["up_kl"]) for x in z)
pad = lambda key, dt: np.stack([np.pad(x[key], [(0, N - len(x[key]))] + [(0, 0)] * (x[key].ndim - 1)) for x in z]).astype(dt)
np.savez(a.out, up=pad("up_tok", np.int32), down=pad("down_tok", np.int32),
         up_kl=pad("up_kl", np.float16), down_kl=pad("down_kl", np.float16))
print(f"{L} layers -> {a.out}")
