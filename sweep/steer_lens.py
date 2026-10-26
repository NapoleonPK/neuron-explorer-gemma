"""Steer lens: what pushing each neuron does to the model's next-token prediction, through every layer after it.

For each neuron and sign, pushes ±dose x the median residual norm along the neuron's unit write column (after the
post-feedforward norm, where the sweep steers; see model_spec.py) at every position of a few short contexts, runs one forward pass, and compares the last position's log-probabilities with the unsteered run.
Keeps, per neuron and sign, the 12 tokens whose log-probability rises most (averaged over contexts) and the KL
divergence from the unsteered prediction.

    uv run python sweep/steer_lens.py --layers 0-7 --out stats/steer_lens
    uv run python sweep/steer_lens.py --out stats/steer_lens                 # every layer

Writes stats/steer_lens_L{l}.npz per layer: up_tok[N, 12], down_tok[N, 12] (token ids for the + push and - push),
up_kl[N], down_kl[N].
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from model_spec import MODEL, chat_ids, layer_scalar, load, text_ids, write_cols

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", default="", help="e.g. 0-7; default every layer (35 on E2B, 42 on E4B, 34 on Gemma 3)")
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default="stats/steer_lens")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = a.device
    model, lm = load(a.model, dev)
    lo, _, hi = (a.layers or f"0-{len(lm.layers) - 1}").partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))
    ctxs = [chat_ids(tok, [{"role": "user", "content": c}]) for c in CHAT] + [text_ids(tok, p) for p in PLAIN]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([[pad] * (T - len(c)) + c for c in ctxs], device=dev)
    mask = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in ctxs], device=dev)
    posn = (mask.cumsum(-1) - 1).clamp(min=0)

    push = {"layer": None, "vecs": None}  # vecs: [rows, hidden], one push per batch row
    resid = {}

    def make_hook(i):
        def hook(mod, inp, out):
            if push["layer"] != i:
                return
            return out + push["vecs"][:, None, :].to(out.dtype)
        return hook

    for i, layer in enumerate(lm.layers):
        layer.post_feedforward_layernorm.register_forward_hook(make_hook(i))
        layer.register_forward_hook(lambda m, inp, out, i=i: resid.__setitem__(i, (out[0] if isinstance(out, tuple) else out)))

    def last_logprobs(input_ids, attn, pos):
        h = lm(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(h).float(), -1)

    with torch.inference_mode():
        base = last_logprobs(ids, mask, posn)  # [C, V]
        # median residual norm leaving each layer over these contexts' real tokens
        norms = {i: r[mask.bool()].float().norm(dim=-1).median().item() for i, r in resid.items()}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    B = a.neurons_per_batch
    for l in layers:
        W = write_cols(lm.layers[l]).float()  # [hidden, this layer's width]
        N = W.shape[1]
        U = (W / W.norm(dim=0)).T  # [N, hidden] unit write columns
        out = {k: [] for k in ("up_tok", "down_tok", "up_kl", "down_kl")}
        t0 = time.time()
        for s in range(0, N, B):
            n = torch.arange(s, min(s + B, N), device=dev)
            b = len(n)
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                # rows: neuron-major, contexts repeated
                cols = n.repeat_interleave(C)
                push.update(layer=l, vecs=sign * a.dose * norms[l] / layer_scalar(lm.layers[l]) * U[cols])
                with torch.inference_mode():
                    lp = last_logprobs(ids.repeat(b, 1), mask.repeat(b, 1), posn.repeat(b, 1)).view(b, C, -1)
                push["layer"] = None
                delta = (lp - base[None]).mean(1)  # [b, V]
                kl = (lp.exp() * (lp - base[None])).sum(-1).mean(1)  # KL(steered || base), averaged over contexts
                out[f"{key}_tok"].append(delta.topk(a.k).indices.int().cpu())
                out[f"{key}_kl"].append(kl.cpu())
        arrs = {k: torch.cat(v).numpy() for k, v in out.items()}
        np.savez(f"{a.out}_L{l}.npz", **arrs, dose=a.dose, contexts=np.array(CHAT + PLAIN))
        print(f"L{l}: {N} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(arrs['up_kl']):.3f}", flush=True)


if __name__ == "__main__":
    main()
