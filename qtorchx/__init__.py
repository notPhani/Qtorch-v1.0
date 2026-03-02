__version__ = "1.0.0"

# Expose the core components for easy access
from .core.primitives import Circuit, Gate, GateLibrary
from .core.backend import QtorchBackend
from .noise.qnaf import PhiManifoldExtractor
from .noise.calibrator import NoiseCalibrator
from .noise.presets import Preset, PresetManager

__all__ = [
    "Circuit",
    "Gate",
    "GateLibrary",
    "QtorchBackend",
    "PhiManifoldExtractor",
    "NoiseCalibrator",
    "Preset",
    "PresetManager"
]