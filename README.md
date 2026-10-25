# neuron explorer — Gemma 3 4B

A port of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), a steering-first neuron
explorer built for Qwen3.5-4B, to **Gemma 3 4B** (`google/gemma-3-4b-it`: 34 layers, 10,240 GeGLU MLP neurons per
layer, hidden size 2560). Every MLP neuron in a layer is pushed up and down on a few prompts, every rollout is kept, and
the explorer lays the results out over a 3D view of the model. All credit for the explorer, the method and the original
findings goes to the original author; this fork only changes what the model switch requires.

![neuron explorer (original, on Qwen3.5-4B)](explorer.png)

## Status

- **Ported and tested end to end on a small randomly initialised Gemma 3** (same architecture, real tokenizer and chat
  template): live steering (add and multiply, multi-turn), activation reads with pushes and head ablation, logprobs,
  logit lens, connections, cones, search, fires, layout, the stats pass, the steer lens and every server endpoint.
- **Not yet run on the real 4B weights or through the vLLM sweep.** No caches or sweep runs ship yet: the Qwen caches
  were removed (they do not fit Gemma's shapes), so the map starts with no sweep data until you run the steps below.
- The Qwen findings in `research_log.md`, `NOTES.md` and the methodology's examples are kept for reference; their
  neuron ids do not carry over to Gemma.

## What changed for Gemma 3

Gemma 3 normalises the MLP branch a second time before adding it to the residual stream (`post_feedforward_layernorm`,
an RMSNorm scaled by 1 + w). So a neuron's `down_proj` column is not the direction it writes into the residual stream.
To first order that is `(1 + w_post) ⊙ W_down[:, n]`, called its **write column** here. All the Gemma specifics live in
`sweep/model_spec.py`:

- **Steering** adds `dose × median residual norm` along the unit write column *after* the post-feedforward norm (the
  vLLM fork's `mlp_out` hook on Gemma 3). On Qwen that was the same as raising the neuron's activation; on Gemma the
  norm would rescale such a raise, so pushes go on the normed branch instead and doses keep their meaning (fractions of
  the residual norm). The sweep now defaults to `--design main_rel --dose-unit rel`. Multiplying and reading a neuron
  still happen at the input of `down_proj`, where `gelu(gate) * up` lives.
- **Logit lens, connections and layouts** use write columns; connections use Gemma's pre-MLP norm
  (`pre_feedforward_layernorm`).
- **Attention**: every layer is softmax attention with 8 query / 4 kv heads; layers 5, 11, 17, 23 and 29 are global,
  the rest use a 1024-token sliding window. The map draws global layers as the thicker slabs.
- **Chat and tokens**: Gemma's chat template, `<end_of_turn>` as the stop token, and `<bos>` at position 0 of every
  packed window (Gemma's residual stream blows up without it).
- **Grid**: 10,240 neurons do not form a square, so each layer's sheet is 102 cells wide with a short last row.
- The vision tower is dropped after loading; the explorer only reads text.
- `explorer/build_steer_index.py` (new) stacks the per-layer steer lens files into `explorer/cache/steer_index.npz`.

## Model access

`google/gemma-3-4b-it` is gated: accept the license on its Hugging Face page and log in (`hf auth login`). Or point
every script at an ungated copy of the same weights:

    export EXPLORER_MODEL=unsloth/gemma-3-4b-it

The live model needs about 9 GB of memory in bf16. On a 16 GB Mac with other apps open that leaves little headroom;
`EXPLORER_NO_MODEL=1` runs the explorer from caches only.

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

On first start with the live model, the server builds `explorer/cache/vocab.json`, the logit-lens index and the
activation ranges (from FineWeb-Edu). The rest of the caches come from:

    uv run python explorer/build_connections.py        # direct-path links, CUDA or MPS
    uv run python explorer/build_layout.py --kind writes
    uv run python sweep/steer_lens.py --layers 0-33 --out stats/steer_lens && \
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
wires the `mlp_in` / `mlp_out` hooks on Gemma 3.

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 17 --tokens 2000000 --out stats/g3-4b
    python sweep/check_equivalence.py --stats stats/g3-4b_L17.npz   # HF vs vLLM steering, before a full sweep
    python sweep/steer_sweep.py --stats stats/g3-4b_L17.npz --neurons all --random-controls 256 --run g3-4b-L17
    sweep/run_depth.sh "5 11 17 23" 384 64                    # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt. The coherent window was found on Qwen; check it on Gemma
with `--design calib` before a full sweep.

| script | output |
|---|---|
| `sweep/model_spec.py` | the model id, file prefix and Gemma 3 specifics every script shares |
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
author credited above; ask the original author before reusing the code outside GitHub. Gemma 3 weights are under
[Google's Gemma terms](https://ai.google.dev/gemma/terms).
