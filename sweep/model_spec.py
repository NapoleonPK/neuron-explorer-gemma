"""The model this explorer is built for, and the Gemma 3 specifics every script shares.

Gemma 3 normalises the MLP branch a second time before adding it to the residual stream (post_feedforward_layernorm,
an RMSNorm that scales by 1 + w). So neuron n's down_proj column is not the direction it writes into the residual
stream; to first order that is (1 + w_post) * W_down[:, n], here called its write column. Additive steering pushes
along the write column after that norm (vLLM's `mlp_out` hook), and the logit lens, connections and layouts all use
it. Multiplying or reading a neuron still happens at the input of down_proj, where gelu(gate) * up lives.

Gemma also needs <bos> at position 0: without it the residual stream blows up, so every packed window starts with it.

    EXPLORER_MODEL=unsloth/gemma-3-4b-it      # an ungated copy of the same weights, if the Google repo is not unlocked
"""
import json
import os

MODEL = os.environ.get("EXPLORER_MODEL", "google/gemma-3-4b-it")
TAG = "g3-4b"  # file prefix in stats/ and runs/


def write_cols(layer):
    """[hidden, inter] write columns of an HF Gemma 3 decoder layer, in the weights' dtype."""
    W = layer.mlp.down_proj.weight
    return W * (1.0 + layer.post_feedforward_layernorm.weight.float()).to(W.dtype)[:, None]


def pre_mlp_scale(layer):
    """[hidden] scale of the norm in front of the MLP (pre_feedforward_layernorm), float32."""
    return 1.0 + layer.pre_feedforward_layernorm.weight.float()


def chat_ids(tok, messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def end_ids(tok):
    """Token ids that end a reply: <end_of_turn> and <eos>."""
    return {t for t in (tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id) if t is not None}


def load(model=MODEL, device=None, dtype=None):
    """(full model, text model) with the vision tower dropped: the explorer only reads text."""
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype or torch.bfloat16)
    for name in ("vision_tower", "multi_modal_projector"):
        if hasattr(m.model, name):
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
        key = next(k for k in self.index if k.endswith("layers.0.mlp.down_proj.weight") and "vision" not in k)
        self.prefix = key[: -len("0.mlp.down_proj.weight")]  # e.g. "language_model.model.layers."

    def tensor(self, layer, name):
        from safetensors import safe_open
        full = f"{self.prefix}{layer}.{name}"
        with safe_open(os.path.join(self.root, self.index[full]), "pt") as f:
            return f.get_tensor(full)

    def write_cols(self, layer):
        """[hidden, inter] float32 write columns, as write_cols() above."""
        W = self.tensor(layer, "mlp.down_proj.weight").float()
        return W * (1.0 + self.tensor(layer, "post_feedforward_layernorm.weight").float())[:, None]
