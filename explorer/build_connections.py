"""Direct-path connections for every neuron in every layer, for the explorer's details tab.

For layers i < j, P[B, A] = (W_gate_j[B] . v_A) * (W_up_j[B] . v_A) with v_A = s_j * c_A, c_A neuron A's write column
and s_j the scale of layer j's pre-MLP norm (sweep/model_spec.py). Neuron A's downstream links in layer j are column A
of P, |P| z-scored over B; neuron B's upstream links in layer i are row B of P, |P| z-scored over A. Each neuron keeps
its k strongest links per side across all layers, with sign(P).

    uv run python explorer/build_connections.py            # CUDA, about 5 min on a 4090

Writes explorer/cache/connections.npz: {down,up}_layer [L, N, k] int8, {down,up}_n int16, {down,up}_z float16,
{down,up}_sign int8, N the widest layer (a narrower layer's padding rows keep z = -60000, i.e. no links).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep"))
from model_spec import MODEL, Checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL)
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "connections.npz"))
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "mps"

ck = Checkpoint(a.model)
L, N, K = len(ck.widths), max(ck.widths), a.k
bf = lambda t: t.to(dev, torch.bfloat16)
down = [bf(ck.write_cols(l)) for l in range(L)]  # write columns, [hidden, N]
gate = [bf(ck.tensor(l, "mlp.gate_proj.weight")) for l in range(L)]
up = [bf(ck.tensor(l, "mlp.up_proj.weight")) for l in range(L)]
norm = [bf(ck.norm_scale(l, "pre_feedforward_layernorm")) for l in range(L)]

# running top-k per neuron and side: score, source layer, source neuron, sign
best = {s: (torch.full((L, N, K), -1e9, device=dev), torch.zeros((L, N, K), dtype=torch.int16, device=dev),
            torch.zeros((L, N, K), dtype=torch.int16, device=dev), torch.zeros((L, N, K), dtype=torch.int8, device=dev))
        for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, sign):
    bz, bl, bn, bs = best[side]
    w = z.shape[0]  # layer l's width
    cz = torch.cat([bz[l, :w], z], 1)
    cl = torch.cat([bl[l, :w], torch.full_like(idx, other_layer, dtype=torch.int16)], 1)
    cn = torch.cat([bn[l, :w], idx.to(torch.int16)], 1)
    cs = torch.cat([bs[l, :w], sign], 1)
    top = cz.topk(K, dim=1)
    bz[l, :w], bl[l, :w], bn[l, :w], bs[l, :w] = (top.values, cl.gather(1, top.indices), cn.gather(1, top.indices),
                                                  cs.gather(1, top.indices))


t0 = time.time()
with torch.inference_mode():
    for j in range(1, L):
        for i in range(j):
            V = down[i] * norm[j][:, None]  # [hidden, N_i]
            P = (gate[j] @ V).float() * (up[j] @ V).float()  # [N_j (B), N_i (A)]
            M = P.abs()
            # downstream of A (layer i): column A over B
            zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
            top = zc.topk(K, dim=0)  # [K, N_i]
            sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
            merge("down", i, top.values.T, top.indices.T, j, sgn.T)
            # upstream of B (layer j): row B over A
            zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
            top = zr.topk(K, dim=1)  # [N_j, K]
            sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
            merge("up", j, top.values, top.indices, i, sgn)
            del P, M, zc, zr
        print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, (bz, bl, bn, bs) in best.items():
    out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
    out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int8)
    out[f"{side}_n"] = bn.cpu().numpy()
    out[f"{side}_sign"] = bs.cpu().numpy()
np.savez_compressed(a.out, **out, k=K)
print("->", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")
