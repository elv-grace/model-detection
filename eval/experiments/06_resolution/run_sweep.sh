set -e
R=/home/elv-grace/model-detection
cd "$R"
P="podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
   --volume=$R/.cache:/root/.cache --volume=$R/eval:/elv/eval \
   --device nvidia.com/gpu=0 --network host -e HF_HOME=/root/.cache \
   -e ELV_WEIGHTS_DIR=/root/.cache/detection localhost/general_detection"
OUT=/elv/eval/experiments/06_resolution

# Ultralytics backends: imgsz must be a multiple of 32. 640 is the trained size.
for SZ in 640 960 1280; do
  echo "=== ultralytics imgsz $SZ ==="
  $P /elv/eval/tools/run_bp.py --backends yoloe26-text yoloe11-text world-text yolo11 \
     --imgsz $SZ --out $OUT/runs_ul_$SZ
done

# HF backends: gdino's stock shortest_edge is 800, owlv2's square side is 960. 800 is the
# baseline rung so the sweep includes each model's own default.
for SZ in 800 1100 1400; do
  echo "=== hf imgsz $SZ ==="
  $P /elv/eval/tools/run_bp.py --backends gdino owlv2 \
     --hf-imgsz $SZ --out $OUT/runs_hf_$SZ
done
echo DONE
