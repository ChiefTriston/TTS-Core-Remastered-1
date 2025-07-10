# reference_encoder/setup.py
from setuptools import setup, find_packages

setup(
    name="reference_encoder",
    version="0.1.0",
    description="Few‐shot speaker reference encoder for TTS-Core-Remastered",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "torchaudio>=2.5.1",
        "transformers>=4.30.0",
        "speechbrain>=0.5.0",
        "librosa",
        "soundfile",
        "pydub",
        "scikit-learn",        # for EER computation
        "numpy",
        "tensorboard",         # for logging
        "tqdm"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "refenc-train=reference_encoder.train:train",
            "refenc-latency=reference_encoder.latency_benchmark:main",
            "refenc-eer=reference_encoder.verify_eer:main",
            "refenc-fuse=reference_encoder.verify_fusion:main",
        ],
    },
)
