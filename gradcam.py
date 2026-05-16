import numpy as np
import cv2
import torch
import torch.nn.functional as F

from config import CONFIG


def _densenet_forward_no_inplace(self_densenet, x: torch.Tensor) -> torch.Tensor:
    """
    Monkey-patch for DenseNet's forward pass that disables the in-place ReLU.
    Required so that Grad-CAM backward hooks can capture gradients correctly.
    """
    features = self_densenet.features(x)
    out = F.relu(features, inplace=False)        # inplace=False is the key fix
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    return self_densenet.classifier(out)


def generate_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    img_size: int = CONFIG['IMG_SIZE'],
) -> np.ndarray:
    """
    Produce a Grad-CAM heatmap for a given class index.

    Parameters
    ----------
    model            : HybridModel (or any model with model.backbone.features.norm5)
    input_tensor     : (1, C, H, W) tensor, already on the correct device.
    target_class_idx : Index of the class to visualise.
    img_size         : Output heatmap size (square).

    Returns
    -------
    cam : np.ndarray of shape (img_size, img_size), values in [0, 1].
    """
    model.eval()

    features_blobs: list  = []
    gradients_blobs: list = []

    def save_feature(module, inp, output):
        features_blobs.append(output.cpu().detach())

    def save_gradient(module, grad_in, grad_out):
        gradients_blobs.append(grad_out[0].cpu().detach())

    target_layer = model.backbone.features.norm5
    handle_f = target_layer.register_forward_hook(save_feature)
    handle_g = target_layer.register_full_backward_hook(save_gradient)

    # Apply the in-place fix temporarily
    original_forward = model.backbone.forward
    model.backbone.forward = _densenet_forward_no_inplace.__get__(
        model.backbone, type(model.backbone)
    )

    # Forward + backward
    output = model(input_tensor.to(CONFIG['DEVICE']))
    score  = output[0, target_class_idx]
    model.zero_grad()
    score.backward()

    # Restore original forward
    model.backbone.forward = original_forward
    handle_f.remove()
    handle_g.remove()

    # Compute weighted combination of feature maps
    grads   = gradients_blobs[0].numpy()
    fmaps   = features_blobs[0].numpy()
    weights = np.mean(grads, axis=(2, 3))[0, :]

    cam = np.zeros(fmaps.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmaps[0, i, :, :]

    # Post-process
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_size, img_size))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
