# neuron explorer — Gemma 4 E2B / E4B (and Gemma 3 4B)

A port of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), a steering-first neuron
explorer built for Qwen3.5-4B, to **Gemma 4 E2B** (`google/gemma-4-E2B-it`: 35 layers, hidden size 1536, 6,144 GeGLU
MLP neurons on layers 0-14 and 12,288 on layers 15-34, about 10 GB in bf16, so it runs on a 16 GB Mac). **Gemma 4 E4B**
(`google/gemma-4-E4B-it`: 42 layers, 10,240 neurons per layer, hidden size 2560) and **Gemma 3 4B**
(`google/gemma-3-4b-it`: 34 layers, 10,240 neurons) are supported too. Every MLP
neuron in a layer is pushed up and down on a few prompts, every rollout is kept, and the explorer lays the results out
over a 3D view of the model. All credit for the explorer, the method and the original findings goes to the original
author; this fork only changes what the model switch requires.

![neuron explorer (original, on Qwen3.5-4B)](explorer.png)

## Status

- **Ported and tested end to end on small randomly initialised Gemma 4 and Gemma 3 models** (same architectures, real
  tokenizers and chat templates; the Gemma 4 one has per-layer embeddings, KV sharing, wide global heads and
  non-trivial layer scalars): live steering (add and multiply, multi-turn), activation reads with pushes and head
  ablation, logprobs, logit lens, connections, cones, search, fires, layouts, the stats pass, the steer lens and every
  server endpoint, plus a Gemma 4 variant with E2B's double-wide layers. Tensor names, shapes, norm weights and layer
  scalars were checked against the real E4B and E2B checkpoints' headers.
- **Not yet run on the real weights or through the vLLM sweep.** No caches or sweep runs ship yet: the Qwen caches
  were removed (they do not fit Gemma's shapes), so the map starts with no sweep data until you run the steps below.
- The Qwen findings in `research_log.md`, `NOTES.md` and the methodology's examples are kept for reference; their
  neuron ids do not carry over to Gemma.

## Choosing the model

    export EXPLORER_MODEL=google/gemma-4-E2B-it     # default, not gated
    export EXPLORER_MODEL=google/gemma-4-E4B-it     # not gated
    export EXPLORER_MODEL=google/gemma-3-4b-it      # gated: accept the license on Hugging Face and `hf auth login`
    export EXPLORER_MODEL=unsloth/gemma-3-4b-it     # an ungated copy of the Gemma 3 weights

Every script reads it. Stats and runs are named after the model (`stats/gemma-4-e2b_L17.npz`, `runs/gemma-4-e2b-L17`),
so models never mix; the caches in `explorer/cache/` belong to one model at a time.

Memory: Gemma 4 E2B's text model is about 9.4 GB in bf16 (4.7 GB of it is the per-layer embedding table), which fits a
16 GB Mac. E4B's is about 15 GB (16 GB checkpoint), so it wants a GPU with 24 GB or more, or a Mac with 32 GB or more.
Gemma 3 4B needs about 9 GB. `EXPLORER_NO_MODEL=1` runs the
explorer from caches only.

## What changed for Gemma

Gemma normalises the MLP branch a second time before adding it to the residual stream (`post_feedforward_layernorm`).
So a neuron's `down_proj` column is not the direction it writes into the residual stream. To first order that is
`s ⊙ W_down[:, n]`, with `s` the norm's scale (`w` on Gemma 4, `1 + w` on Gemma 3), called its **write column** here.
All the model specifics live in `sweep/model_spec.py`:

- **Steering** adds `dose × median residual norm` along the unit write column *after* the post-feedforward norm (the
  vLLM fork's `mlp_out` hook on Gemma). On Qwen that was the same as raising the neuron's activation; on Gemma the
  norm would rescale such a raise, so pushes go on the normed branch instead and doses keep their meaning (fractions of
  the residual norm). The sweep now defaults to `--design main_rel --dose-unit rel`. Multiplying and reading a neuron
  still happen at the input of `down_proj`, where `gelu(gate) * up` lives.
- **Gemma 4 layer scalars**: each layer multiplies its whole output by a learned scalar (0.06 on layer 0, about 0.4
  elsewhere in E4B), so pushes are sent divided by it and still move the residual leaving the layer by the intended
  amount. After the MLP, Gemma 4 also adds a per-layer token embedding gated by the residual stream; that gate sees
  the push too, so a push's total effect on the layer output differs a little from its nominal size, as it would for a
  real change in the neuron's activation.
- **Logit lens, connections and layouts** use write columns; connections use Gemma's pre-MLP norm
  (`pre_feedforward_layernorm`).
- **Attention**: every layer is softmax attention with 8 query heads. Some layers are global (4, 9, … 34 on E2B; 5, 11,
  … 41 on E4B; 5, 11, 17, 23, 29 on Gemma 3), the rest use a sliding window (512 tokens on Gemma 4, 1024 on Gemma 3).
  Gemma 4's global heads are 512 wide, and its last layers reuse earlier layers' keys and values (20 on E2B, 18 on
  E4B). The map draws global layers as the thicker slabs.
- **Chat and tokens**: each model's own chat template and end-of-turn token (`<turn|>` on Gemma 4, `<end_of_turn>` on
  Gemma 3), and `<bos>` at position 0 of every read and packed window (Gemma's residual stream blows up without it, and
  Gemma 4's tokenizer no longer adds it to plain text).
- **Layer widths**: E2B's KV-shared layers (15-34) have double-wide MLPs, 12,288 neurons against 6,144. Arrays over
  every neuron of every layer (activations, lens and steer indexes, connections, layouts) are padded to the widest
  layer; `mlp_widths()` in `sweep/model_spec.py` gives each layer's real width, the server masks the padding, and the
  map leaves it out, so the narrower sheets fill only their top rows.
- **Grid**: each sheet is ceil(sqrt(N)) cells wide for the widest layer's N (111 on E2B, 102 on E4B and Gemma 3), with a
  short last row.
- The vision and audio parts are dropped after loading; the explorer only reads text.
- `explorer/build_steer_index.py` (new) stacks the per-layer steer lens files into `explorer/cache/steer_index.npz`.
- Endpoints that need the connection index (cone, web) answer with an empty result and a note until
  `build_connections.py` has run.

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

On first start with the live model, the server builds `explorer/cache/vocab.json`, the logit-lens index and the
activation ranges (from FineWeb-Edu). The rest of the caches come from:

    uv run python explorer/build_connections.py        # direct-path links, CUDA or MPS
    uv run python explorer/build_layout.py --kind writes
    uv run python sweep/steer_lens.py --out stats/steer_lens && \
        uv run python explorer/build_steer_index.py --stats stats/steer_lens

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar write columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## Sweep

Runs on a CUDA GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)), which
wires the `mlp_in` / `mlp_out` hooks on Gemma 3 and Gemma 4. A 24-48 GB card is plenty for E2B; for E4B, a 40-80 GB
card (A100, H100, L40S) leaves room for the 16 GB of weights plus a large batch of rollouts. `neuron_stats.py` and
`steer_lens.py` also run on a Mac (MPS).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 17 --tokens 2000000
    python sweep/check_equivalence.py --stats stats/gemma-4-e2b_L17.npz   # HF vs vLLM steering, before a full sweep
    python sweep/steer_sweep.py --stats stats/gemma-4-e2b_L17.npz --neurons all --random-controls 256 --run gemma-4-e2b-L17
    sweep/run_depth.sh "4 9 14 19 24 29 34" 384 64              # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt. The coherent window was found on Qwen; check it on Gemma
with `--design calib` before a full sweep.

| script | output |
|---|---|
| `sweep/model_spec.py` | the model id, file prefix and Gemma specifics every script shares |
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down and write columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `explorer/build_steer_index.py` | `explorer/cache/steer_index.npz` from the steer lens files |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the original project's findings and reasoning (on Qwen3.5-4B); `NOTES.md` the source notes.
`scripts/` holds the original project's research scripts, unchanged.

## License

The upstream repository has no license file. This fork is published through GitHub's fork feature with the original
author credited above; ask the original author before reusing the code outside GitHub. Gemma weights come under
Google's own terms (see each model's Hugging Face page).
