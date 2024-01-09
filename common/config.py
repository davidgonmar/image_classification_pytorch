import os
import torch

config_dir = os.path.dirname(os.path.abspath(__file__))


SAVED_PATH = os.path.join(config_dir, "..", "saved")
DATA_PATH = os.path.join(config_dir, "..", "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
