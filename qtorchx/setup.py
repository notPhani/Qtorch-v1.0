from setuptools import setup, find_packages
import os

setup(
    name="qtorchx",
    version="1.0.0",
    author="Vikram",
    description="A Differentiable Quantum Noise Simulator powered by the QNaF 7D Manifold",
    long_description=open("README.md").read() if "README.md" in os.listdir() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # Ensures .qnaf files in the presets folder are bundled
    package_data={
        "qtorchx.noise": ["presets/*.qnaf"],
    },
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "hardware": ["qiskit>=1.0.0", "qiskit-ibm-runtime"],
    },
    python_requires=">=3.8",
)