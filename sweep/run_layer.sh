#!/usr/bin/env bash
# Stats pass then a full-layer sweep for one layer. Logs to stats_L<layer>.log and sweep_<run>.log; detach with
#     setsid nohup sweep/run_layer.sh 8 gemma-4-e4b-L8 </dev/null >/dev/null 2>&1 &
# Extra arguments pass through to steer_sweep.py.
set -euo pipefail
LAYER="$1"; RUN="$2"; shift 2
cd /workspace/neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
TAG=$(python -c "import sys; sys.path.insert(0, 'sweep'); from model_spec import TAG; print(TAG)")  # follows EXPLORER_MODEL
mkdir -p stats runs
[ -f "stats/${TAG}_L${LAYER}.npz" ] || python sweep/neuron_stats.py --layers "$LAYER" --tokens 2000000 \
    --out stats/${TAG} > "stats_L${LAYER}.log" 2>&1
python sweep/steer_sweep.py --stats "stats/${TAG}_L${LAYER}.npz" --design main_rel --dose-unit rel --neurons all \
    --random-controls 256 --max-num-seqs 320 --run "$RUN" "$@" > "sweep_${RUN}.log" 2>&1
echo SWEEP_DONE >> "sweep_${RUN}.log"
