import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import seaborn as sns
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import CONFIG, CHESTX14_LABELS
from gradcam import generate_gradcam


# ──────────────────────────────────────────────
#  Core Evaluation
# ──────────────────────────────────────────────
def evaluate_model(model, loader: DataLoader, device) -> tuple:
    """
    Run inference on *loader* and collect predictions + ground-truth.

    Returns
    -------
    y_pred : np.ndarray  (N, NUM_CLASSES) – sigmoid probabilities
    y_true : np.ndarray  (N, NUM_CLASSES) – binary labels
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            outputs = torch.sigmoid(model(imgs.to(device)))
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_preds), np.vstack(all_labels)


def print_auc_scores(y_pred: np.ndarray, y_true: np.ndarray) -> tuple:
    """Print per-class ROC-AUC scores and return valid scores + label names."""
    print("\n" + "=" * 40)
    print("FINAL METRICS (CLASS-WISE)")
    print("=" * 40)

    auc_scores, valid_labels = [], []
    for i, label in enumerate(CHESTX14_LABELS):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(score)
            valid_labels.append(label)
            print(f"{label:20} | ROC-AUC: {score:.3f}")
        except ValueError:
            print(f"{label:20} | ROC-AUC: N/A (only one class in y_true)")

    return auc_scores, valid_labels


# ──────────────────────────────────────────────
#  Disease Co-occurrence Heatmap
# ──────────────────────────────────────────────
def plot_disease_cooccurrence(df, labels=CHESTX14_LABELS):
    """Plot a normalised co-occurrence heatmap for the given disease labels."""
    sns.set_style("whitegrid")

    disease_df = df[labels]
    cooccurrence = disease_df.T.dot(disease_df)

    disease_counts = disease_df.sum(axis=0)
    normalised = cooccurrence.div(disease_counts.values + 1e-8, axis=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        normalised,
        annot=True,
        cmap='viridis',
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Co-occurrence Probability'},
    )
    plt.title('Disease Co-occurrence Heatmap', fontsize=16)
    plt.xlabel('Disease', fontsize=12)
    plt.ylabel('Disease', fontsize=12)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────
def denormalize(tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return HWC numpy array in [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(std * img + mean, 0, 1)


# ──────────────────────────────────────────────
#  Quick Result Visualisation
# ──────────────────────────────────────────────
def plot_results(sample_img, y_pred: np.ndarray, model, auc_scores, valid_labels):
    """
    Three-panel figure:
      Left  – original X-ray bar chart of per-class AUC
      Centre – original X-ray image
      Right  – Grad-CAM overlay for the top predicted class
    """
    idx        = 0
    probs      = y_pred[idx]
    target_idx = np.argmax(probs)

    heatmap = generate_gradcam(model, sample_img.unsqueeze(0), target_idx)
    orig    = denormalize(sample_img)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.barh(valid_labels, auc_scores, color='skyblue')
    plt.title("Pathology AUC Scores")
    plt.xlim(0, 1)

    plt.subplot(1, 3, 2)
    plt.imshow(orig, cmap='bone')
    plt.title("Original X-ray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(orig, cmap='bone')
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.title(f"Grad-CAM: {CHESTX14_LABELS[target_idx]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
