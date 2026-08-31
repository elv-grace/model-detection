#!/bin/bash
#
# Local smoke test for the general_detection container.
#
# Mirrors buildscripts/testers/test-model.sh (the canonical `make test` harness) but is a
# thin, editable wrapper to vary fps easily:
#
#   ./test.sh          # fps=1 (default)
#   ./test.sh 5        # fps=5
#
# Unlike the runtime's own invocation, file paths are fed on STDIN (run_default reads
# stdin, not argv), --output-path is required, and tunables like fps go in as a --params
# JSON string. The detection_cache named volume is mounted at /root/.cache, covering both
# HF_HOME (SigLIP 2) and storage.cache_path (YOLOE + MobileCLIP), so the several GB of
# weights download once and are reused across runs.

# No `set -e`: we capture the container's exit code below and still dump output on failure.
set -uo pipefail

# fps from $1 (default 1); which GPU from ELV_MODEL_TEST_GPU_TO_USE (default 3).
FPS="${1:-1}"
: "${ELV_MODEL_TEST_GPU_TO_USE:=3}"
IMAGE_NAME="${IMAGE_NAME:-general_detection}"

cd "$(dirname "$0")"

set -x

rm -rf test-output/
mkdir -p test-output

# Container-side paths (test-files/ is bind-mounted read-only at /elv/test), newline-separated on stdin.
INPUT=$(find test-files/ -maxdepth 1 -type f | sed 's|^test-files/|/elv/test/|' | sort)

echo "$INPUT" | podman run --rm -i \
    --volume="$(pwd)/test-files:/elv/test:ro" \
    --volume="$(pwd)/test-output:/elv/tags:U" \
    --volume=detection_cache:/root/.cache \
    --network host \
    --device "nvidia.com/gpu=${ELV_MODEL_TEST_GPU_TO_USE}" \
    "${IMAGE_NAME}" \
    --output-path /elv/tags/out.jsonl \
    --params "{\"fps\": ${FPS}}"

ex=$?

set +x
echo "=== test-output/out.jsonl ==="
find test-output -type f

exit $ex
