set -e
R=/home/elv-grace/model-detection
cd "$R"
P="podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
   --volume=$R/.cache:/root/.cache --volume=$R/eval:/elv/eval \
   --device nvidia.com/gpu=0 --network host -e HF_HOME=/root/.cache \
   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   -e ELV_WEIGHTS_DIR=/root/.cache/detection localhost/general_detection"
OUT=/elv/eval/experiments/06_resolution
# Batch drops as resolution rises: activation memory goes with the square of the input.
for SPEC in "800 4" "1100 2" "1400 1"; do
  set -- $SPEC
  echo "=== hf imgsz $1 batch $2 ==="
  $P /elv/eval/tools/run_bp.py --backends gdino owlv2 \
     --hf-imgsz $1 --batch $2 --out $OUT/runs_hf_$1
done
echo DONE
