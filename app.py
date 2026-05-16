"""
app.py – Gradio-based Clinical AI Diagnostic Dashboard
Run with:  python app.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import gradio as gr
from PIL import Image
from torchvision import transforms, models

from config import CONFIG, CHESTX14_LABELS

# ──────────────────────────────────────────────
#  Image Transforms
# ──────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────
#  Model (standalone re-definition for app.py)
# ──────────────────────────────────────────────
class HybridModel(nn.Module):
    """DenseNet-121 + Transformer encoder for multi-label classification."""

    def __init__(self, num_classes: int = CONFIG['NUM_CLASSES']):
        super().__init__()
        backbone = models.densenet121(weights='DEFAULT')
        num_features = backbone.classifier.in_features

        # Expose features block directly (needed for Grad-CAM hook)
        self.features    = backbone.features
        self.projection  = nn.Linear(num_features, 512)

        encoder_layer    = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier  = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        out   = F.adaptive_avg_pool2d(feats, (1, 1))
        out   = torch.flatten(out, 1)
        proj  = self.projection(out).unsqueeze(1)
        trans = self.transformer(proj).squeeze(1)
        return self.classifier(trans)


# ──────────────────────────────────────────────
#  Grad-CAM
# ──────────────────────────────────────────────
def get_gradcam(model: HybridModel, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
    model.eval()
    target_layer = model.features.norm5

    feature_maps: list = []
    gradients:    list = []

    h1 = target_layer.register_forward_hook(lambda m, i, o: feature_maps.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class_idx].backward()

    grads   = gradients[0].cpu().data.numpy()
    fmaps   = feature_maps[0].cpu().data.numpy()
    weights = np.mean(grads, axis=(2, 3))[0, :]

    cam = np.zeros(fmaps.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmaps[0, i, :, :]

    h1.remove()
    h2.remove()

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ──────────────────────────────────────────────
#  Device & Model Initialisation
# ──────────────────────────────────────────────
DEVICE = CONFIG['DEVICE']
model  = HybridModel().to(DEVICE)
model.eval()

# To use a trained checkpoint, uncomment the next line:
# model.load_state_dict(torch.load("hybrid_model.pth", map_location=DEVICE))


# ──────────────────────────────────────────────
#  Prediction Function
# ──────────────────────────────────────────────
def predict_chest_xray(img: np.ndarray):
    """
    Gradio-compatible prediction function.

    Parameters
    ----------
    img : np.ndarray  (H, W, 3) uint8 from Gradio image component.

    Returns
    -------
    results      : dict  {label: probability}
    overlayed    : np.ndarray  Grad-CAM overlay image
    """
    if img is None:
        return None, None

    img_pil      = Image.fromarray(img).convert('RGB')
    input_tensor = IMG_TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.set_grad_enabled(True):          # needed for Grad-CAM
        outputs = model(input_tensor)
        probs   = torch.sigmoid(outputs).detach().cpu().numpy()[0]

    results       = {CHESTX14_LABELS[i]: float(probs[i]) for i in range(len(CHESTX14_LABELS))}
    top_class_idx = int(np.argmax(probs))
    heatmap       = get_gradcam(model, input_tensor, top_class_idx)

    # Overlay heatmap on original image
    img_res           = np.array(img_pil.resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])))
    heatmap_colored   = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored   = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlayed         = cv2.addWeighted(img_res, 0.6, heatmap_colored, 0.4, 0)

    return results, overlayed


# ──────────────────────────────────────────────
#  Gradio UI
# ──────────────────────────────────────────────
CSS = """
.gradio-container {
    background: url('https://scitechdaily.com/images/DNA-Genetics.gif') !important;
    background-size: cover !important;
    background-attachment: fixed !important;
}
.glass-card {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    color: white !important;
}
#clinical-header { text-align: center; color: #3b82f6 !important; }
.glass-card span, .glass-card label, .glass-card p { color: white !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    with gr.Column(elem_classes="glass-card"):
        gr.Markdown("# 🏥 Clinical AI Diagnostic Dashboard", elem_id="clinical-header")
        gr.Markdown("### Automated Pulmonary Pathology Localisation Engine", elem_id="clinical-header")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Patient Scan Upload", type="numpy")
                with gr.Row():
                    btn_run   = gr.Button("⚡ Analyse Image", variant="primary")
                    btn_clear = gr.Button("🔄 Reset")

            with gr.Column(scale=1):
                output_labels = gr.Label(num_top_classes=5, label="Detection Probabilities")
                output_cam    = gr.Image(label="AI Diagnostic Focus (Heatmap)")

        gr.Markdown("🟢 **System Status:** Ready for diagnostic analysis.")

    btn_run.click(fn=predict_chest_xray, inputs=input_img, outputs=[output_labels, output_cam])
    btn_clear.click(lambda: (None, None), None, [input_img, output_cam])


if __name__ == "__main__":
    demo.launch(debug=True)
