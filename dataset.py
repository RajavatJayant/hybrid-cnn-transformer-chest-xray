import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from config import CONFIG, CHESTX14_LABELS


# ──────────────────────────────────────────────
#  Image Index (built once at startup)
# ──────────────────────────────────────────────
def build_image_index(data_dir: str) -> dict:
    """Recursively index all PNG images in data_dir → {filename: full_path}."""
    print("Indexing image paths… this may take a minute.")
    index = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(data_dir, '**/*.png'), recursive=True)
    }
    print(f"Found {len(index)} images.")
    return index


# ──────────────────────────────────────────────
#  Data Preparation
# ──────────────────────────────────────────────
def prepare_data(csv_file: str):
    """
    Load the NIH metadata CSV, encode multi-label columns,
    and split by patient ID to prevent data leakage.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    df = pd.read_csv(csv_file)

    # Encode each disease label as a binary column
    for label in CHESTX14_LABELS:
        df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)

    # Patient-level split (avoids same patient appearing in train & test)
    unique_patients = df['Patient ID'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )

    train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
    test_df  = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)
    return train_df, test_df


# ──────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset compatible with NIH ChestX-14."""

    def __init__(self, df: pd.DataFrame, image_index: dict, transform=None):
        """
        Parameters
        ----------
        df          : DataFrame with 'Image Index' column and label columns.
        image_index : Dict mapping filename → full path (from build_image_index).
        transform   : torchvision transforms to apply.
        """
        self.df          = df
        self.image_index = image_index
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self.image_index[row['Image Index']]
        image    = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[CHESTX14_LABELS].values.astype(float), dtype=torch.float32
        )
        return image, labels
