import os
import torch

# ──────────────────────────────────────────────
#  Dataset & Training Configuration
# ──────────────────────────────────────────────
CONFIG = {
    'DATA_DIR': None,           # Set at runtime via kagglehub
    'CSV_FILE': None,           # Set at runtime
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'NUM_CLASSES': 14,
    'LR': 1e-4,
    'EPOCHS': 10,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# ──────────────────────────────────────────────
#  Disease Labels (NIH ChestX-14)
# ──────────────────────────────────────────────
CHESTX14_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]
