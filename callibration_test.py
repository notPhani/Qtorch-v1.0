import torch
from qtorchx.core.primitives import Circuit, Gate
from qtorchx.noise.presets import PresetManager
from qtorchx.noise.qnaf import PhiManifoldExtractor
from qtorchx.core.backend import QtorchBackend
from qtorchx.noise.calibrator import NoiseCalibrator

def test_full_workflow():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Testing QtorchX Workflow on {device}")

    # --- PHASE 1: STANDARD SIMULATION ---
    print("\n[Path A] Running Standard Simulation...")
    circ = Circuit(2)
    circ.add(Gate('H', [0]))
    circ.add(Gate('CNOT', [0, 1]))

    # Load Bundled Preset
    preset = PresetManager.fetch("qtorch_local", device=device)
    extractor = PhiManifoldExtractor(circ, preset.dpm, preset.bpo, device=device)
    noisy_circ = extractor.annotate_circuit()

    # Execute
    backend = QtorchBackend(circuit=noisy_circ, simulate_with_noise=True, device=device)
    results = backend.get_histogram_data(shots=100)
    print(f"   Counts: {results}")

    # --- PHASE 2: CALIBRATION ---
    print("\n[Path B] Building Digital Twin from Randomness...")
    
    # 1. Define Target (What we want to converge to)
    # Let's say we want to match a 10% bit-flip on Q0
    target_probs = torch.zeros(3, 2, noisy_circ.depth, device=device)
    target_probs[0, 0, :] = 0.10 

    # 2. Initialize with Random Preset Copy
    calibrator = NoiseCalibrator(circ, preset_name="qtorch_standard", device=device)
    with torch.no_grad():
        calibrator.dpm.copy_(torch.randn(3, 6, device=device) * 0.1)

    # 3. Train (Backtracking)
    history = calibrator.calibrate(target_probs, epochs=5000)
    print(f"   Final Loss: {history[-1]:.6f}")

    # 4. Export & Execute
    calibrator.export_qnaf("trained_twin")
    print("   Digital Twin exported as 'trained_twin.qnaf'")

if __name__ == "__main__":
    test_full_workflow()