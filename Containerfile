FROM continuumio/miniconda3:latest
WORKDIR /elv

# transformers>=5 / torch 2.8 require Python 3.11+
RUN conda create -n mlpod python=3.11 -y

# ffmpeg provides the ffprobe/ffmpeg CLIs used by common_ml.video_processing (and the
# codecs PyAV decodes video with); build-essential for any source builds.
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification (common-ml is a git dependency)
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

# The torch 2.8 pip wheels bundle their own CUDA runtime, so no host CUDA toolkit is
# needed; the container just needs the NVIDIA driver + container toolkit at run time
# (--device nvidia.com/gpu=...).

# Install dependencies before copying source so the heavy dependency layer is cached.
# `pip install .` installs the dependencies (incl. common-ml from git); the tagger code
# itself runs from the source copied below (WORKDIR is on sys.path).
COPY setup.py .
RUN mkdir -p general_detection
RUN /opt/conda/envs/mlpod/bin/pip install .

# No weights are baked into this image. Two separate downloads happen on first load:
#   - SigLIP 2 from the HuggingFace hub, into HF_HOME
#   - the YOLOE checkpoint and the MobileCLIP text encoder that get_text_pe() needs,
#     into storage.cache_path (config.yml); ultralytics resolves those relative to the
#     CWD, which general_detection/detector.py handles by chdir-ing into the cache during load.
# Both live under /root/.cache, so ONE mounted volume there covers both. Without it they
# land in the container's ephemeral writable layer and are re-fetched on every run.
ENV HF_HOME=/root/.cache
# Keep ultralytics' settings/config out of the ephemeral layer too.
ENV YOLO_CONFIG_DIR=/root/.cache/Ultralytics

# Source last (changes most often).
COPY general_detection ./general_detection
COPY config.yml run.py config.py ./

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "-u", "run.py"]
