import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import kagglehub

from config import CONFIG, CHESTX14_LABELS
from dataset import build_image_index, prepare_data, ChestXrayDataset
from model import HybridModel


# ──────────────────────────────────────────────
#  Standard ImageNet transforms
# ──────────────────────────────────────────────
def get_transforms(img_size: int = CONFIG['IMG_SIZE']) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
#  Training Entry Point
# ──────────────────────────────────────────────
def train(max_batches: int = None):
    """
    Train the HybridModel on the NIH ChestX-14 dataset.

    Parameters
    ----------
    max_batches : int, optional
        If set, stop training early after this many batches (useful for quick tests).
    """
    # 1. Download / locate dataset
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    print("Dataset path:", path)

    CONFIG['DATA_DIR'] = path
    CONFIG['CSV_FILE'] = f"{path}/Data_Entry_2017.csv"

    # 2. Build image index and prepare splits
    image_index         = build_image_index(CONFIG['DATA_DIR'])
    train_df, test_df   = prepare_data(CONFIG['CSV_FILE'])

    # 3. DataLoaders
    tfm         = get_transforms()
    train_set   = ChestXrayDataset(train_df, image_index, tfm)
    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 4. Model, loss, optimizer
    device    = CONFIG['DEVICE']
    model     = HybridModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'])

    # 5. Training loop
    model.train()
    print("Starting training…")
    for epoch in range(CONFIG['EPOCHS']):
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] "
                      f"Batch {batch_idx:4d} | Loss: {loss.item():.4f}")

        print(f"── Epoch {epoch+1} complete ──")

    # 6. Save checkpoint
    torch.save(model.state_dict(), "hybrid_model.pth")
    print("Model saved → hybrid_model.pth")
    return model


if __name__ == "__main__":
    train()
