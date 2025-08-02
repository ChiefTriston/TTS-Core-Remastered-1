# EmotionDriftObserver_v3.1/setup.py

from setuptools import setup, find_packages

setup(
    name="emotion_drift_observer",
    version="3.1.0",
    description="Emotion Drift Observer with advanced speaker diarization and analysis",
    packages=find_packages(include=["emotion_drift_observer", "emotion_drift_observer.*"]),
    install_requires=[
        "torch>=1.10.0",
        "torchaudio",
        "numpy",
        "scikit-learn",
        "librosa",
        "speechbrain",
        "resemblyzer",
        "pyyaml",
        "networkx",
        "bidict",
        "faiss-cpu",  # Use faiss-gpu if CUDA is preferred
        "webrtcvad",
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
        "console_scripts": [
            "emotiondriftobserver=emotion_drift_observer.hyper_diarizer:main",
        ]
    },
)