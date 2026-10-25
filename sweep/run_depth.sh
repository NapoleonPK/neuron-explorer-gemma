#!/usr/bin/env bash
# Depth survey for a list of layers: one short stats pass covering them all, then a random-subset sweep per layer.
#     setsid nohup sweep/run_depth.sh "0 4 9 13" 384 64 </dev/null >/dev/null 2>&1 &
set -uo pipefail
LAYERS="$1"; NNEUR="${2:-384}"; NCTRL="${3:-64}"
cd /workspace/neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
TAG=$(python -c "import sys; sys.path.insert(0, 'sweep'); from model_spec import TAG; print(TAG)")  # follows EXPLORER_MODEL
mkdir -p stats runs
need=""; for l in $LAYERS; do [ -f "stats/${TAG}_L$l.npz" ] || need="$need,$l"; done
if [ -n "$need" ]; then
  python sweep/neuron_stats.py --layers "${need#,}" --tokens 200000 --out stats/${TAG} > depth_stats.log 2>&1
fi
for l in $LAYERS; do
  echo "L$l start $(date)" >> depth.log
  python sweep/steer_sweep.py --stats "stats/${TAG}_L$l.npz" --design main_rel --dose-unit rel --neurons "random:$NNEUR" \
    --random-controls "$NCTRL" --max-num-seqs 320 --run "${TAG}-L$l-r$NNEUR" > "sweep_L$l-r$NNEUR.log" 2>&1 \
    && echo "L$l done $(date)" >> depth.log || echo "L$l FAILED $(date)" >> depth.log
done
echo ALL_DONE >> depth.log
