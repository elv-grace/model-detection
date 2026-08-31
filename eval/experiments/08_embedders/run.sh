set -e
R=/home/elv-grace/model-detection
cd "$R"
podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
  --volume=$R/.cache:/root/.cache --volume=$R/eval:/elv/eval \
  --volume=/ml/pools/logo_pool:/ml/pools/logo_pool:ro \
  --device nvidia.com/gpu=0 --network host -e HF_HOME=/root/.cache \
  localhost/general_detection /elv/eval/tools/compare_embedders.py \
  --brands 1500 --per-brand 5 \
  --json-out /elv/eval/experiments/08_embedders/scores.json
echo DONE
