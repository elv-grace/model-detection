from setuptools import setup

setup(
    name="general-detection",
    version="0.1",
    packages=["general_detection"],
    python_requires=">=3.11",
    # This package links ultralytics (AGPL-3.0); see LICENSE.
    license="AGPL-3.0-or-later",
    install_requires=[
        # model + inference
        'torch==2.8.*',
        # torchvision matched to torch 2.8: the SigLIP 2 checkpoints ship a fast image
        # processor built on torchvision transforms, and ultralytics needs it too.
        'torchvision==0.23.*',
        # >=5 because `Siglip2ImageProcessor` names the torchvision-backed fast processor
        # in 5.x and the numpy/PIL one in 4.x. Their resampling differs slightly, so two
        # workers on opposite sides of that boundary would write subtly different vectors
        # into the same index.
        'transformers>=5.0.0',
        'accelerate>=1.12.0',
        # YOLOE (open-vocabulary detection). AGPL-3.0 — see LICENSE.
        'ultralytics>=8.3.150',
        # Detections container + the per-class NMS used for the synonym-group dedupe pass.
        'supervision>=0.26.0',
        'Pillow>=10.0.0',
        'numpy',
        # image read path in common_ml.tagging.file_tagger (images tagged frame-directly)
        'opencv-python-headless>=4.12.0.88',
        # video frame extraction: common_ml.video_processing decodes/samples frames with
        # PyAV. Also imported unconditionally at startup by common_ml.tagging.run_helpers.
        'av',
        # tagger runtime / plumbing
        'loguru==0.5.2',
        'setproctitle',
        'dacite',
        'ujson',
        'tqdm',
        'pyyaml',
        'common-ml @ git+https://github.com/eluv-io/common-ml@vector-tags',
    ],
    extras_require={
        'test': ['pytest'],
    },
)
