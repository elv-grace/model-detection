#!/bin/bash

set -e

git submodule update --init --recursive

# Both models (SigLIP 2 from the HuggingFace hub, YOLOE from ultralytics' assets) are
# pulled at runtime into a mounted cache, so there are no baked-in weights to sync before
# building.
exec buildscripts/build_container.bash -t "general_detection:${IMAGE_TAG:-latest}" . -f Containerfile
