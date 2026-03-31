"""Setup script for the STT engine."""

from setuptools import setup, find_packages


setup(
    name="stt-engine",
    version="0.1.0",
    description="Custom Speech-to-Text Engine using Conformer-CTC",
    author="STT Project",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "jiwer>=3.0.0",
    ],
    extras_require={
        "api": ["fastapi>=0.100.0", "uvicorn>=0.23.0", "python-multipart>=0.0.6"],
        "lm": ["pyctcdecode>=0.5.0", "kenlm"],
        "export": ["onnx>=1.14.0", "onnxruntime>=1.15.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "stt-train=scripts.train:main",
            "stt-transcribe=scripts.transcribe:main",
            "stt-evaluate=scripts.evaluate:main",
            "stt-api=scripts.run_api:main",
        ],
    },
)
