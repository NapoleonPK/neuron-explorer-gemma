"""Per-neuron activation ranges for MLP layers of Gemma 3, plus their down-projection and write columns.

A neuron's activation is gelu(gate) * up, the input to down_proj. Positive activations come from two sign regimes
(gate>0, up>0) and (gate<0, up<0); negative ones from (gate>0, up<0) and (gate<0, up>0). This records the extreme of
each regime per neuron over a mixed web + chat corpus, and saves every neuron's write column (W_down[:, n] scaled by
the post-feedforward norm, see model_spec.py), which is the direction a steering push on neuron n adds to the residual
stream.

    uv run python sweep/neuron_stats.py --layers 17 --tokens 2000000 --out stats/g3-4b

Writes stats/g3-4b_L17.npz with
    pp_max, nn_max   largest positive activation in the (+,+) and (-,-) regimes          [I]
    pn_min, np_min   most negative activation in the (gate+, up-) and (gate-, up+) regimes [I]
    abs_max          max |activation| over everything (the unit for steering doses)      [I]
    w_down           down_proj columns, one row per neuron                                [I, H] float16
    w_write          write columns, one row per neuron (what steering pushes along)        [I, H] float16
    resid_norm_q10_50_90   quantiles of the per-token residual norm leaving the layer (scale for push sizes)
"""
import argparse
import itertools
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model_spec import MODEL, TAG, load, write_cols


def text_stream(tok, seed):
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        yield tok.apply_chat_template(c["messages"], tokenize=False).removeprefix(tok.bos_token)


def windows(tok, seq_len, batch, seed):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows, each starting with <bos> (Gemma needs it
    at position 0; it is also the sink position the stats skip)."""
    buf = []
    eos, n = tok.eos_token_id, (seq_len - 1) * batch
    for text in text_stream(tok, seed):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= n:
            chunk, buf = buf[:n], buf[n:]
            body = torch.tensor(chunk).view(batch, seq_len - 1)
            yield torch.cat([torch.full((batch, 1), tok.bos_token_id), body], 1)


class RegimeStats:
    def __init__(self, n, device):
        f = lambda v: torch.full((n,), v, device=device, dtype=torch.float32)
        self.pp_max, self.nn_max = f(0.0), f(0.0)
        self.pn_min, self.np_min = f(0.0), f(0.0)
        self.gate = self.up = None
        self.resid_norms = []  # per-token residual norm leaving the layer, positions >= 1 (position 0 is a sink)

    def update_resid(self, out):
        h = out[0] if isinstance(out, tuple) else out
        self.resid_norms.append(h[:, 1:].float().norm(dim=-1).flatten().cpu())

    def update(self, act):
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        g, u, a = (t[:, 1:].flatten(0, 1).float() for t in (self.gate, self.up, act))
        gp, up_ = g > 0, u > 0
        ninf = torch.tensor(float("-inf"), device=a.device)
        pinf = torch.tensor(float("inf"), device=a.device)
        self.pp_max = torch.maximum(self.pp_max, torch.where(gp & up_, a, ninf).amax(0))
        self.nn_max = torch.maximum(self.nn_max, torch.where(~gp & ~up_, a, ninf).amax(0))
        self.pn_min = torch.minimum(self.pn_min, torch.where(gp & ~up_, a, pinf).amin(0))
        self.np_min = torch.minimum(self.np_min, torch.where(~gp & up_, a, pinf).amin(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", default="17", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"stats/{TAG}")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(a.model)
    _, lm = load(a.model, a.device)
    lm.layers = lm.layers[: max(layers) + 1]  # nothing past the deepest target layer is needed

    stats = {}
    for l in layers:
        mlp = lm.layers[l].mlp
        s = stats[l] = RegimeStats(mlp.down_proj.in_features, a.device)
        mlp.gate_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "gate", o))
        mlp.up_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "up", o))
        mlp.down_proj.register_forward_pre_hook(lambda m, i, s=s: s.update(i[0]))
        lm.layers[l].register_forward_hook(lambda m, i, o, s=s: s.update_resid(o))

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    t0 = time.time()
    with torch.inference_mode():
        for step, ids in enumerate(itertools.islice(windows(tok, a.seq_len, a.batch, a.seed), n_steps)):
            lm(input_ids=ids.to(a.device), use_cache=False)
            if step % 10 == 0 or step == n_steps - 1:
                done = (step + 1) * a.seq_len * a.batch
                print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for l, s in stats.items():
        pp, nn, pn, np_ = (x.cpu().numpy() for x in (s.pp_max, s.nn_max, s.pn_min, s.np_min))
        abs_max = np.maximum.reduce([pp, nn, -pn, -np_])
        w_down = lm.layers[l].mlp.down_proj.weight.detach().float().T.cpu().numpy().astype(np.float16)
        w_write = write_cols(lm.layers[l]).detach().float().T.cpu().numpy().astype(np.float16)
        resid_q = np.quantile(torch.cat(s.resid_norms).numpy(), [0.1, 0.5, 0.9])
        path = f"{a.out}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down, w_write=w_write,
                 resid_norm_q10_50_90=resid_q, model=a.model, layer=l, tokens=n_steps * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}")
        dn_top = (nn > pp).mean()
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {dn_top:.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
