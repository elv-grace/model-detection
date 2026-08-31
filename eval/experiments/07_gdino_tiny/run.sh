set -e
R=/home/elv-grace/model-detection
cd "$R"
P="podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
   --volume=$R/.cache:/root/.cache --volume=$R/eval:/elv/eval \
   --device nvidia.com/gpu=0 --network host -e HF_HOME=/root/.cache \
   -e ELV_WEIGHTS_DIR=/root/.cache/detection localhost/general_detection"
$P /elv/eval/tools/run_bp.py --backends gdino-tiny --batch 4 \
   --out /elv/eval/experiments/07_gdino_tiny/runs
echo DONE
