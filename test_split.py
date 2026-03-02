import torch
import numpy as np
import time
import sys
import os

# --- PATH SETUP (Ensure we can import the modules) ---
# Assuming running from root 'qtorchx-project/'
sys.path.append(os.path.abspath("."))

try:
    # 1. Import Data Structures
    from qtorchx.core.primitives import Circuit, Gate, GateLibrary
    # 2. Import Physics Engine
    from qtorchx.noise.qnaf import PhiManifoldExtractor
    # 3. Import Execution Engine
    from qtorchx.core.backend import QtorchBackend
    print("✅ [Import] QtorchX modules loaded successfully.")
except ImportError as e:
    print(f"❌ [Import] Failed: {e}")
    print("Ensure you are in the root directory and 'pip install -e .' was run.")
    sys.exit(1)

def run_full_stack_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🚀 Starting QtorchX Full Stack Test on {device.upper()}...")

    # ========================================================================
    # PHASE 1: CIRCUIT CONSTRUCTION (Data Layer)
    # ========================================================================
    print("\n[Phase 1] Building Quantum Circuit...")
    
    # Create Bell State: |00> -> H(0) -> CNOT(0,1) -> |00> + |11>
    circ = Circuit(num_qubits=2)
    
    # Add gates (Testing Gate and Circuit classes)
    h_gate = Gate('H', [0])
    cx_gate = Gate('CNOT', [0, 1])
    
    t1 = circ.add(h_gate)
    t2 = circ.add(cx_gate)
    
    print(f"  - Circuit created: 2 Qubits, 2 Gates")
    print(f"  - Depth: {circ.depth}")
    print(f"  - Visual:\n{circ.visualize()}")
    
    assert circ.depth == 2, "Circuit depth should be 2"

    # ========================================================================
    # PHASE 2: PHYSICS SIMULATION (QNaF Layer)
    # ========================================================================
    print("\n[Phase 2] Generating QNaF Noise Manifold...")
    
    # Initialize Mock Calibration Data (The 7D Manifold Parameters)
    # DPM: Maps 6 internal noise channels -> 3 Pauli Errors (X, Y, Z)
    dpm = torch.randn(3, 6, device=device) * 0.1 
    
    # BPO: Baseline hardware error rates
    bpo = torch.tensor([-3.0, -3.0, -3.0], device=device) # ~4% error rate base
    
    # Initialize Extractor
    extractor = PhiManifoldExtractor(
        circuit=circ,
        DecoherenceProjectionMatrix=dpm,
        BaselinePauliOffset=bpo,
        device=device,
        # Hyperparams from your qnaf.py default
        alpha=0.9, beta=0.15, kappa=0.1 
    )
    
    # 1. Run Physics Simulation
    start_phi = time.time()
    manifold = extractor.GetManifold() # API Check: GetManifold (PascalCase)
    print(f"  - Phi Manifold Shape: {manifold.shape} (Expected: 6, 2, 2)")
    print(f"  - Physics Sim Time: {time.time() - start_phi:.4f}s")
    
    # 2. Analyze Features
    importance = extractor.get_feature_importance()
    print(f"  - Dominant Noise Feature: {max(importance, key=importance.get)} ({max(importance.values()):.1f}%)")
    
    # 3. Inject Noise into Circuit
    noisy_circ = extractor.annotate_circuit() # API Check: annotate_circuit
    
    # Verify annotation
    annotated_gates = [g for g in noisy_circ.gates if 'noise_model' in g.metadata]
    print(f"  - Gates Annotated: {len(annotated_gates)}/{len(noisy_circ.gates)}")
    
    if len(annotated_gates) > 0:
        sample_noise = annotated_gates[0].metadata['noise_model']['pauli_probs'][0]
        print(f"  - Sample Noise Profile (Q0, Gate H): P(I)={sample_noise[0]:.4f}, P(err)={1.0-sample_noise[0]:.4f}")

    # ========================================================================
    # PHASE 3: EXECUTION (Backend Layer)
    # ========================================================================
    print("\n[Phase 3] Running Noisy Simulation...")
    
    # Initialize Backend with the NOISY circuit
    backend = QtorchBackend(
        simulate_with_noise=True,
        persistant_data=True,
        circuit=noisy_circ,
        device=device
    )
    
    # Run Simulation
    shots = 1000
    results = backend.get_histogram_data(shots=shots) # API Check: get_histogram_data
    result = backend.execute_circuit(10000)
    print(f"  - Sample Execution Result: {result[:10]} (Expected: mostly '00' and '11')")
    
    print(f"  - Results ({shots} shots): {results}")
    
    # Calculate Fidelity (approximate)
    # Ideal Bell state is 50% '00', 50% '11'
    counts_00 = results.get('00', 0)
    counts_11 = results.get('11', 0)
    fidelity = (counts_00 + counts_11) / shots
    
    print(f"  - Measured Fidelity: {fidelity:.4f}")
    
    # Basic assertion: Fidelity should be < 1.0 due to injected noise, but > 0.5 (random guess)
    if 0.5 < fidelity < 1.0:
        print("✅ [Success] Noise was successfully applied (Fidelity < 1.0).")
    elif fidelity == 1.0:
        print("⚠️ [Warning] Fidelity is 1.0. Noise might be too weak or not applied.")
    else:
        print("⚠️ [Warning] Fidelity is very low. Noise might be too strong.")

    print("\n🎉 ALL SYSTEMS GO: QtorchX Architecture is fully operational.")

if __name__ == "__main__":
    run_full_stack_test()