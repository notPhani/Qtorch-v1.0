from setuptools import setup, find_packages
import os

# Read README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qtorchx",
    version="1.0.0",
    author="Vikram",
    description="A Differentiable Quantum Noise Simulator powered by the QNaF 7D Manifold",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # Ensures .qnaf files in the presets folder are bundled
    package_data={
        "qtorchx.noise": ["presets/*.qnaf"],
    },
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",  # Generic torch to support both CPU and CUDA
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "cuda": ["torch>=2.0.0"], # Users can run 'pip install qtorchx[cuda]'
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)