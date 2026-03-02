# QtorchX: High-Fidelity Quantum Simulation Engine

### Direct-to-Hardware Digital Twin Calibration via the QNaF Formalism

**QtorchX** is a research-grade quantum simulation framework engineered to bridge the gap between ideal computational models and the stochastic realities of physical hardware.

Instead of injecting memoryless noise as independent coin flips, QtorchX models the quantum processor as a dynamical physical environment governed by a hidden **7D noise manifold**. Built entirely on **PyTorch**, the platform provides a fully differentiable pipeline from gate execution to measurement statistics.

In short:

> QtorchX doesn’t sprinkle noise on circuits.
> It evolves it.

This enables a powerful inverse workflow: backtracking from empirical hardware histograms to the latent physical parameters of the noise field — creating a high-precision **Digital Twin** of the quantum device.

---

# The QNaF Physical Formalism

At the core of QtorchX lies the **QNaF (Quantum Noise as Fields) Framework**.

Rather than assigning static Pauli error probabilities, QNaF defines a structured, time-evolving manifold:

[
\Phi(t) \in \mathbb{R}^{7 \times Q \times T}
]

Where:

* ( Q ) = number of qubits
* ( T ) = circuit depth
* 7 = independent decoherence channels

The manifold evolves according to a nonlinear, topology-aware dynamical system:

[
\Phi(t+1) =
\tanh\Big(
\mathcal{M}\Phi(t)

* \beta L \Phi(t)
* \kappa L D(t)
* \mathcal{N}(\Phi(t))
* \Sigma(t)
  \Big)
  ]

Where:

* ( L ) = graph Laplacian derived from circuit connectivity
* ( D(t) ) = gate-induced disturbance field
* ( \mathcal{M} ) = temporal memory operator
* ( \mathcal{N} ) = nonlinear coupling term
* ( \Sigma(t) ) = stochastic environmental fluctuations

The tanh activation ensures bounded thermodynamic stability:

[
\Phi \in [-1, 1]
]

Noise is no longer independent per gate — it diffuses, accumulates, saturates, and interacts across qubits and time.

---

## Learnable Decoherence Projections

The hidden manifold does not directly corrupt the quantum state. Instead, it is projected into Pauli error space via two trainable tensors:

[
P(t) = W \Phi(t) + B
]

Where:

* ( W \in \mathbb{R}^{3 \times 7} ) → **Decoherence Projection Matrix (DPM)**
* ( B \in \mathbb{R}^{3} ) → **Baseline Pauli Offset (BPO)**

The resulting Pauli channel strengths:

[
p_{X,Y,Z}(q,t) = \sigma(P_{X,Y,Z}(q,t))
]

Where ( \sigma ) is a sigmoid mapping into valid probability space.

Both **DPM** and **BPO** are fully differentiable and trainable.

This enables:

* Hardware fingerprint discovery
* Device-specific dephasing dominance modeling
* Automatic calibration via gradient descent

In effect, QtorchX learns how a processor decoheres.

---

# Technical Architecture

QtorchX is structured into three execution layers optimized for GPU-native performance and zero-copy tensor flow.

---

## 1️⃣ Core Simulation Engine (`qtorchx.core`)

The backend is a statevector engine designed to avoid the (2^n \times 2^n) matrix expansion bottleneck.

Instead of expanding operators globally, QtorchX performs:

[
|\psi\rangle \rightarrow \text{reshape} \rightarrow \text{permute} \rightarrow U \rightarrow \text{inverse-permute}
]

This guarantees:

* O((2^n)) memory scaling
* Linear depth scaling
* Native GPU acceleration via PyTorch

### Intelligent Caching

* Static gates → O(1) lookup
* Parametric gates → LRU cache with angle quantization
* Optional persistent gate fusion

The result: realistic noise modeling with minimal constant overhead.

---

## 2️⃣ Physics Layer (`qtorchx.noise`)

The **PhiManifoldExtractor** evolves the hidden noise field at each time slice of the circuit.

Each gate modifies the disturbance field:

[
D_i(t) = \alpha G_i(t) + \beta M_i(t)
]

Where:

* ( G_i(t) ) = gate activity
* ( M_i(t) ) = measurement activity

Noise becomes a function of circuit structure — not an afterthought.

---

## 3️⃣ Calibration Layer (`qtorchx.noise.calibrator`)

The **NoiseCalibrator** closes the loop.

Given empirical bitstring probabilities ( \hat{p} ), the system minimizes:

[
\mathcal{L} =
\text{MSE}(p_{\text{sim}}, \hat{p})

* \lambda \cdot \text{StabilityPenalty}
  ]

Gradients flow through:

* Statevector execution
* Manifold projection
* DPM and BPO tensors

Result:

[
\frac{\partial \mathcal{L}}{\partial W},
\quad
\frac{\partial \mathcal{L}}{\partial B}
]

This enables direct-to-hardware calibration.

---

# 🛠 API Implementation Guide

## High-Fidelity Simulation

```python
import qtorchx as qtx

circ = qtx.Circuit(num_qubits=5)
circ.add('H', [0]).add('CNOT', [0, 1])

preset = qtx.PresetManager.fetch("qtorch_local")

backend = qtx.QtorchBackend(circuit=circ, simulate_with_noise=True)
results = backend.get_histogram_data(shots=1000)
```

---

## Real Hardware Calibration

```python
calibrator = qtx.NoiseCalibrator(circ, preset_name="qtorch_standard")

history = calibrator.calibrate(target_data, epochs=100)

calibrator.export_qnaf("calibrated_twin_v1")
```

After calibration, the exported `.qnaf` preset becomes a digital twin of the target hardware.

---

# Deployment & Credits

**Version**: 1.0.0
**Lead Engineer**: Phani Kumar
**Status**: Stable Release (Patent Pending)

### Installation

```bash
pip install qtorchx
```

For optimal calibration throughput, a CUDA-enabled PyTorch installation is strongly recommended.

---

QtorchX is licensed under the MIT License.
The QNaF formalism and associated differentiable architectures are protected research assets.

---
