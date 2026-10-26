"""The model this explorer is built for, and the Gemma specifics every script shares. Gemma 4 E2B is the default (it
fits a 16 GB Mac); Gemma 4 E4B and Gemma 3 4B work too.

    EXPLORER_MODEL=google/gemma-4-E2B-it     # default, ~10 GB in bf16
    EXPLORER_MODEL=google/gemma-4-E4B-it     # ~16 GB
    EXPLORER_MODEL=google/gemma-3-4b-it      # gated; unsloth/gemma-3-4b-it is an ungated copy of the same weights

Gemma normalises the MLP branch a second time before adding it to the residual stream (post_feedforward_layernorm).
So neuron n's down_proj column is not the direction it writes into the residual stream; to first order that is
s * W_down[:, n], with s the post-feedforward norm's scale (1 + w on Gemma 3, w on Gemma 4), here called its write
column. Additive steering pushes along the write column after that norm (vLLM's `mlp_out` hook), and the logit lens,
connections and layouts all use it. Multiplying or reading a neuron still happens at the input of down_proj, where
gelu(gate) * up lives.

Gemma 4 then multiplies each layer's whole output by a learned scalar (layer_scalar), after the per-layer input
embedding step, so a push meant to move the residual leaving the layer by p is added as p / layer_scalar.

E2B's MLPs are not all the same width: its KV-shared layers (15-34) are double wide, 12288 neurons against 6144
(use_double_wide_mlp). Arrays over every neuron of every layer are padded to the widest layer; mlp_widths() says where
each layer's real neurons end.

Gemma needs <bos> at position 0: without it the residual stream blows up. Gemma 4's tokenizer no longer adds it to
plain text, so text_ids() does, and every packed window starts with it.
"""
import json
import os
import re

MODEL = os.environ.get("EXPLORER_MODEL", "google/gemma-4-E2B-it")
TAG = re.sub(r"-it$", "", re.sub(r"[^a-z0-9]+", "-", os.path.basename(MODEL.rstrip("/")).lower()))  # file prefix


def mlp_widths(cfg):
    """Neurons in each layer's MLP, from a text config (object or dict). Gemma 4 doubles the width of its KV-shared
    layers when use_double_wide_mlp is set (E2B), as transformers' Gemma4TextMLP does."""
    g = cfg.get if isinstance(cfg, dict) else lambda k: getattr(cfg, k, None)
    L, n = g("num_hidden_layers"), g("intermediate_size")
    first = L - (g("num_kv_shared_layers") or 0)
    return [2 * n if g("use_double_wide_mlp") and l >= first > 0 else n for l in range(L)]


def _plain_norm(norm):
    """Gemma 4's RMSNorm scales by w; Gemma 3's (and earlier) by 1 + w."""
    return type(norm).__name__.startswith("Gemma4")


def norm_scale(norm):
    """[hidden] float32 scale an HF Gemma RMSNorm applies after normalising."""
    w = norm.weight.float()
    return w if _plain_norm(norm) else 1.0 + w


def write_cols(layer):
    """[hidden, inter] write columns of an HF Gemma decoder layer, in the weights' dtype."""
    W = layer.mlp.down_proj.weight
    return W * norm_scale(layer.post_feedforward_layernorm).to(W.dtype)[:, None]


def pre_mlp_scale(layer):
    """[hidden] scale of the norm in front of the MLP (pre_feedforward_layernorm), float32."""
    return norm_scale(layer.pre_feedforward_layernorm)


def layer_scalar(layer):
    """The factor a layer multiplies its whole output by (Gemma 4), else 1."""
    s = getattr(layer, "layer_scalar", None)
    return float(s.flatten()[0]) if s is not None else 1.0


def chat_ids(tok, messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def text_ids(tok, text):
    """Token ids of plain text, starting with <bos>."""
    ids = list(tok(text).input_ids)
    return ids if not ids or ids[0] == tok.bos_token_id else [tok.bos_token_id] + ids


def end_ids(tok):
    """Token ids that end a reply: the end-of-turn token (<turn|> on Gemma 4, <end_of_turn> on Gemma 3) and <eos>."""
    ends = {tok.eos_token_id}
    for t in ("<turn|>", "<end_of_turn>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            ends.add(i)
    return ends - {None}


def load(model=MODEL, device=None, dtype=None):
    """(full model, text model) with the vision and audio parts dropped: the explorer only reads text."""
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype or torch.bfloat16)
    for name in ("vision_tower", "multi_modal_projector", "audio_tower", "embed_vision", "embed_audio"):
        if getattr(m.model, name, None) is not None:
            setattr(m.model, name, None)
    if device:
        m = m.to(device)
    return m.eval(), m.model.language_model


class Checkpoint:
    """Reads text-model tensors straight from the safetensors shards, whatever prefix the checkpoint uses."""

    def __init__(self, model=MODEL):
        from huggingface_hub import snapshot_download
        self.root = model if os.path.isdir(model) else snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
        index_path = os.path.join(self.root, "model.safetensors.index.json")
        if os.path.exists(index_path):
            self.index = json.load(open(index_path))["weight_map"]
        else:
            from safetensors import safe_open
            with safe_open(os.path.join(self.root, "model.safetensors"), "pt") as f:
                self.index = {k: "model.safetensors" for k in f.keys()}
        cfg = json.load(open(os.path.join(self.root, "config.json")))
        self.cfg = cfg.get("text_config", cfg)
        self.plain_norm = self.cfg.get("model_type", "").startswith("gemma4")
        key = next(k for k in self.index if k.endswith("layers.0.mlp.down_proj.weight")
                   and not any(s in k for s in ("vision", "audio")))
        self.prefix = key[: -len("0.mlp.down_proj.weight")]  # e.g. "model.language_model.layers."
        self.widths = mlp_widths(self.cfg)

    def tensor(self, layer, name):
        from safetensors import safe_open
        full = f"{self.prefix}{layer}.{name}"
        with safe_open(os.path.join(self.root, self.index[full]), "pt") as f:
            return f.get_tensor(full)

    def norm_scale(self, layer, name):
        w = self.tensor(layer, f"{name}.weight").float()
        return w if self.plain_norm else 1.0 + w

    def write_cols(self, layer):
        """[hidden, inter] float32 write columns, as write_cols() above."""
        W = self.tensor(layer, "mlp.down_proj.weight").float()
        return W * self.norm_scale(layer, "post_feedforward_layernorm")[:, None]
