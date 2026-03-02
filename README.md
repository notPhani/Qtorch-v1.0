---

# QtorchX: High-Fidelity Quantum Simulation Engine

### Direct-to-Hardware Digital Twin Calibration via the QNaF Formalism

**QtorchX** is a research-grade quantum simulation framework engineered to bridge the gap between ideal computational models and the stochastic realities of physical hardware. By replacing conventional, memoryless noise injection with the **QNaF (Quantum Noise as Fields)** formalism, QtorchX enables the creation of high-precision **Digital Twins** of quantum processors.

Unlike traditional simulators that treat errors as independent coin flips, QtorchX recontextualizes the quantum processor as a physical environment governed by a hidden 7D noise manifold. Built entirely on **PyTorch**, the platform offers a fully differentiable pipeline from gate execution to measurement. This allows researchers to solve the inverse problem of quantum characterization: backtracking from empirical hardware counts to the physical parameters of the noise field.

---

## The QNaF Physical Formalism

At the heart of QtorchX is the **QNaF (Quantum Noise Activation Framework)**. In this model, decoherence is not a static probability but a continuous, time-evolving field $\Phi$ that interacts bidirectionally with the circuit topology.

The noise state is represented as a **7-Dimensional Manifold**, where each dimension captures a distinct physical decoherence vector:

* **Temporal Recurrence**: Error states that persist and accumulate across consecutive gate cycles.
* **Topological Diffusion**: Noise leakage across the hardware graph, governed by a precomputed Laplacian.
* **Gate-Induced Disturbance**: Localized field "bursts" triggered by high-energy operations like CNOTs or Measurements.
* **Nonlocal Bleed**: Cross-talk effects that propagate across non-adjacent qubits via exponential decay.
* **Non-linear Coupling & Kicks**: Higher-order manifold interactions and high-frequency environmental fluctuations.

### Learnable Decoherence Projections

The interface between the hidden manifold and the quantum state is defined by two primary trainable tensors: the **Decoherence Projection Matrix (DPM)** and the **Baseline Pauli Offset (BPO)**. The DPM maps the 7-channel manifold into the 3-channel Pauli error space ($X, Y, Z$), while the BPO establishes the quiescent thermal noise baseline of the device.

---

##  Technical Architecture

QtorchX is divided into three functional layers designed for zero-copy execution and differentiable optimization.

### 1. The Core Simulation Engine (`qtorchx.core`)

The backend is a GPU-accelerated statevector engine that utilizes **Tensor-Reshaping** logic to avoid the $2^n \times 2^n$ matrix expansion bottleneck. This ensures that simulation time scales linearly with circuit depth rather than exponentially with gate application.

* **Intelligent Caching**: Static gates utilize O(1) fixed lookups, while parametric gates employ an LRU cache with angle quantization to maximize hit rates.
* **Permutation-Based Logic**: Qubits are manipulated via efficient dimension permutation, ensuring k-qubit gates are applied at native GPU speeds.

### 2. The Physics Layer (`qtorchx.noise`)

The **PhiManifoldExtractor** acts as the dynamical bridge between the circuit and the QNaF equations. It calculates the evolution of the noise field at every gate step, ensuring that noise is a direct consequence of the spatio-temporal structure of the algorithm.

### 3. The Calibration Layer (`qtorchx.noise.calibrator`)

The **NoiseCalibrator** utilizes the PyTorch autograd engine to flow gradients from hardware-observed bitstring frequencies back to the DPM and BPO. This "Backtracking" allows the simulator to autonomously discover the physical noise profile of any QPU given enough experimental data.

---

## 🛠️ API Implementation Guide

### Initializing a High-Fidelity Simulation

Researchers can utilize pre-calibrated hardware fingerprints to run realistic simulations of known hardware environments.

```python
import qtorchx as qtx

# Define the circuit topology
circ = qtx.Circuit(num_qubits=5)
circ.add('H', [0]).add('CNOT', [0, 1])

# Fetch a calibrated dephasing-dominant hardware preset
preset = qtx.PresetManager.fetch("qtorch_local")

# Execute via the GPU backend
backend = qtx.QtorchBackend(circuit=circ, simulate_with_noise=True)
results = backend.get_histogram_data(shots=1000)

```

### Real Hardware Calibration (Backtracking)

The calibration suite allows for the transformation of raw hardware histograms into a local, differentiable simulator.

```python
# target_data: Empirical probabilities from real-world QPU
calibrator = qtx.NoiseCalibrator(circ, preset_name="qtorch_standard")

# Execute gradient descent to align simulator parameters
history = calibrator.calibrate(target_data, epochs=100)

# Export the learned DPM/BPO tensors as a .qnaf fingerprint
calibrator.export_qnaf("calibrated_twin_v1")

```

## 📜 Deployment & Credits

**Version**: 1.0.0

**Lead Engineer**: Phani Kumar, 

**Status**: Stable Release (Patent Pending)

**Installation**:

```bash
pip install qtorchx

```

QtorchX is licensed under the MIT License. The QNaF formalism and associated differentiable architectures are protected research assets.

*Note: For optimal calibration throughput, a CUDA-enabled PyTorch environment is strictly recommended.*
