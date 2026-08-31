set -e
R=/home/elv-grace/model-detection
cd "$R"
pwd
P="podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
   --volume=$R/.cache:/root/.cache --volume=$R/eval:/elv/eval \
   --device nvidia.com/gpu=0 --network host -e HF_HOME=/root/.cache \
   -e ELV_WEIGHTS_DIR=/root/.cache/detection localhost/general_detection"
B="gdino owlv2 yoloe11-text yoloe26-text world-text"
echo "=== mark7 (6 marks + symbol) ==="
$P /elv/eval/tools/run_bp.py --backends $B \
  --prompts logo "letter logo" "car logo" emblem brand label symbol person \
  --out /elv/eval/experiments/05_symbol_ablation/runs_mark7
echo "=== symbol alone ==="
$P /elv/eval/tools/run_bp.py --backends $B \
  --prompts symbol person \
  --out /elv/eval/experiments/05_symbol_ablation/runs_symbol_only
echo DONE
