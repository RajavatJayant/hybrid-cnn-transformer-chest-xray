import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import seaborn as sns

from config import CONFIG, CHESTX14_LABELS
from gradcam import generate_gradcam


class DiagnosticDashboard:
    """
    Interactive diagnostic dashboard that combines:
    - Grad-CAM CNN focus map
    - Transformer attention map (uniform proxy – see note below)
    - Predicted pathology probabilities bar chart
    """

    def __init__(self, model, labels=CHESTX14_LABELS):
        self.model  = model
        self.labels = labels
        self.device = next(model.parameters()).device

    # ──────────────────────────────────────────────
    #  Attention Extraction
    # ──────────────────────────────────────────────
    def extract_transformer_attention(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Attempt to extract a spatial attention map from the Transformer.

        Note: The current HybridModel uses a single global token, so its
        self-attention produces a 1×1 matrix – not spatially meaningful.
        A uniform 7×7 map is returned as a placeholder until multi-token
        attention is implemented.
        """
        self.model.eval()
        with torch.no_grad():
            features_conv   = self.model.backbone.features(img_tensor.to(self.device))
            pooled          = F.adaptive_avg_pool2d(features_conv, (1, 1))
            flattened       = torch.flatten(pooled, 1)
            _proj_features  = self.model.projection(flattened)   # unused but kept for clarity

            # Placeholder: uniform attention map
            uniform = np.ones((7, 7), dtype=np.float32)
            attn_map = cv2.resize(uniform, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            return attn_map

    # ──────────────────────────────────────────────
    #  Main Diagnostic Runner
    # ──────────────────────────────────────────────
    def run_diagnostic(self, dataset, idx: int = 0):
        """
        Generate a full diagnostic figure for a single patient scan.

        Parameters
        ----------
        dataset : ChestXrayDataset
        idx     : Index of the patient to visualise.
        """
        sns.set_style("whitegrid")

        img_tensor, _ = dataset[idx]
        input_batch   = img_tensor.unsqueeze(0).to(self.device)

        # Un-normalised image for display
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_batch)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        top_class_idx = np.argmax(probs)

        # Grad-CAM
        grad_cam = generate_gradcam(self.model, input_batch, top_class_idx)

        # Attention map
        attn_map = self.extract_transformer_attention(input_batch)

        # Sorted probabilities for bar chart
        sorted_idx    = np.argsort(probs)
        sorted_probs  = probs[sorted_idx]
        sorted_labels = [self.labels[i] for i in sorted_idx]

        # ── Figure ──────────────────────────────────
        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.3])

        # Panel 0 – Original X-ray
        ax0 = plt.subplot(gs[0, 0])
        ax0.imshow(img)
        ax0.set_title("Original Chest X-ray", fontsize=14, fontweight='bold')
        ax0.axis('off')

        # Panel 1 – Grad-CAM
        ax1 = plt.subplot(gs[0, 1])
        ax1.imshow(img)
        cam_plot = ax1.imshow(grad_cam, cmap='inferno', alpha=0.45)
        ax1.set_title(f"CNN Focus: {self.labels[top_class_idx]}", fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(cam_plot, ax=ax1, fraction=0.046)

        # Panel 2 – Transformer attention
        ax2 = plt.subplot(gs[1, 0])
        ax2.imshow(img)
        att_plot = ax2.imshow(attn_map, cmap='magma', alpha=0.5)
        ax2.set_title("Transformer Attention", fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(att_plot, ax=ax2, fraction=0.046)

        # Panel 3 – Combined overlay
        ax3 = plt.subplot(gs[1, 1])
        ax3.imshow(img)
        ax3.imshow(grad_cam,  cmap='inferno', alpha=0.35)
        ax3.imshow(attn_map,  cmap='cool',    alpha=0.25)
        ax3.set_title("Combined Attention", fontsize=14, fontweight='bold')
        ax3.axis('off')

        # Panel 4 – Probability bar chart
        ax4    = plt.subplot(gs[:, 2])
        colors = ['crimson' if p > 0.15 else 'steelblue' for p in sorted_probs]
        ax4.barh(sorted_labels, sorted_probs * 100, color=colors)
        ax4.set_title("Predicted Pathology Probabilities", fontsize=16, fontweight='bold')
        ax4.set_xlabel("Confidence (%)")
        ax4.grid(axis='x', linestyle='--', alpha=0.4)
        for i, v in enumerate(sorted_probs * 100):
            ax4.text(v + 1, i, f"{v:.1f}%", va='center')

        plt.tight_layout()
        plt.show()
