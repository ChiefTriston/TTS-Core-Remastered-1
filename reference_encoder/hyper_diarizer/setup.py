import os
from setuptools import setup, find_packages

setup(
    name="hyper_diarizer",
    version="0.1.0",
    description="HyperDiarizer: Advanced speaker diarization with embedding, clustering, overlap detection, and re-identification",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "faiss-cpu",
        "librosa",
        "resemblyzer",
        "webrtcvad",
        "speechbrain",
        "torch>=1.10.0",
        "torchaudio",
        "pyyaml",
        "networkx",
        "bidict",
        "openai-whisper",
        "demucs",
        "pyannote.core>=4.0",
        "pyannote.metrics",
        "matplotlib",
        "plotly",
        "tqdm",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'hyperdiarizer=hyper_diarizer.cli:main',
        ],
    },
)