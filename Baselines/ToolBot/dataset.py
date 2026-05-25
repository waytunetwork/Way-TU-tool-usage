import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

class ToolBotDataset(Dataset):
    """
    PyTorch Dataset for ToolBot:
    - inputs/ contains *_rgb.npy and *_depth.npy
    - outputs/ contains *_hmg.npy (16 bins) and *_hmm.npy (32 bins)
    """
    def __init__(self, root_dir, transform=None):
        self.inputs_dir  = os.path.join(root_dir, "inputs")
        self.outputs_dir = os.path.join(root_dir, "outputs")
        self.transform   = transform

        # List all RGB files and derive keys like "data_0"
        self.rgb_paths = sorted(glob.glob(os.path.join(self.inputs_dir, "*_rgb.npy")))
        self.keys = [os.path.basename(p)[:-len("_rgb.npy")] for p in self.rgb_paths]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        # Load numpy arrays
        rgb = np.load(os.path.join(self.inputs_dir,  f"{key}_rgb.npy"))    # (H,W,3) uint8
        depth = np.load(os.path.join(self.inputs_dir,  f"{key}_depth.npy")) # (H,W)   float32
        hmg  = np.load(os.path.join(self.outputs_dir, f"{key}_hmg.npy"))    # (16,H,W) float32
        hmm  = np.load(os.path.join(self.outputs_dir, f"{key}_hmm.npy"))    # (32,H,W) float32

        # To torch tensors
        rgb_t   = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)            # (1,H,W)
        hmg_t   = torch.from_numpy(hmg.astype(np.float32))                           # (16,H,W)
        hmm_t   = torch.from_numpy(hmm.astype(np.float32))                           # (32,H,W)

        sample = {
            "rgb": rgb_t,
            "depth": depth_t,
            "heatmap_grasp": hmg_t,
            "heatmap_manip": hmm_t
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

