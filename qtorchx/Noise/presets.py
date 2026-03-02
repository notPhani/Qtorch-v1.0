from dataclasses import dataclass
import torch
import json
import os

@dataclass
class Preset:
    """
    The .qnaf file structure. 
    Encapsulates a specific hardware's noise fingerprint.
    """
    name: str
    dpm: torch.Tensor
    bpo: torch.Tensor
    is_trainable: bool = True
    metadata: dict = None

    def save(self, folder_path: str):
        """Saves the preset as a .qnaf (JSON-based) file."""
        file_path = os.path.join(folder_path, f"{self.name}.qnaf")
        data = {
            "name": self.name,
            "dpm": self.dpm.detach().cpu().tolist(),
            "bpo": self.bpo.detach().cpu().tolist(),
            "is_trainable": self.is_trainable,
            "metadata": self.metadata or {}
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Preset saved: {file_path}")

    @classmethod
    def load(cls, file_path: str, device: str = 'cpu'):
        """Loads a .qnaf file back into a Preset object."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data["name"],
            dpm=torch.tensor(data["dpm"], device=device),
            bpo=torch.tensor(data["bpo"], device=device),
            is_trainable=data["is_trainable"],
            metadata=data["metadata"]
        )