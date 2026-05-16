# Hybrid CNN-Transformer with Dual Explainability for Multi-Label Chest X-Ray Disease Detection

> **Final Year Group Project** | B.Tech / AI & ML  
> Academic Year: 2025–2026

---

## 📌 Abstract

Chest X-ray interpretation is one of the most critical yet cognitively demanding tasks in clinical radiology. This project presents a hybrid deep learning framework that combines a **Convolutional Neural Network (DenseNet-121)** with a **Transformer Encoder** for accurate multi-label classification of 14 pulmonary pathologies from chest X-rays. To address the black-box nature of deep learning in medical imaging, the system incorporates **dual explainability** through **Grad-CAM** (CNN-level spatial heatmaps) and **Transformer Attention Maps**, providing clinicians with visual justification for each prediction. An interactive **Gradio-based diagnostic dashboard** enables real-time inference and visual analysis.

---

## 🧠 Model Architecture

```
Input Chest X-Ray (224 × 224)
          │
    ┌─────▼──────┐
    │ DenseNet-121│  ← Pretrained CNN Backbone (ImageNet)
    │  Backbone   │
    └─────┬───────┘
          │  Feature Maps (B, 1024, 7, 7)
          │  Global Average Pooling → (B, 1024)
          │
    ┌─────▼───────┐
    │   Linear    │  ← Projection Layer: 1024 → 512
    │  Projection │
    └─────┬───────┘
          │
    ┌─────▼────────────────┐
    │  Transformer Encoder  │  ← 2 Layers, 8 Attention Heads
    │   (Self-Attention)    │
    └─────┬─────────────────┘
          │
    ┌─────▼───────┐
    │   Linear    │  ← Multi-Label Classifier: 512 → 14
    │  Classifier │
    └─────┬───────┘
          │
   14 Disease Probabilities
```

---

## 🔍 Dual Explainability

| Method | What it Shows |
|---|---|
| **Grad-CAM** | Spatial regions of the X-ray the CNN focused on |
| **Transformer Attention** | Global token-level attention weights from the encoder |
| **Combined Overlay** | Both maps fused on the original scan for richer clinical insight |

---

## 📁 Project Structure

```
hybrid-cnn-transformer-chest-xray/
│
├── config.py          # Hyperparameters, disease labels, device config
├── dataset.py         # ChestXrayDataset + patient-level data split
├── model.py           # HybridModel: DenseNet-121 + Transformer Encoder
├── gradcam.py         # Grad-CAM heatmap generation (CNN explainability)
├── train.py           # Training loop with AdamW optimizer + checkpoint saving
├── evaluate.py        # ROC-AUC metrics + disease co-occurrence heatmap
├── dashboard.py       # DiagnosticDashboard — multi-panel matplotlib figure
├── app.py             # Gradio web application (real-time inference + heatmap)
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 📊 Dataset — NIH ChestX-14

- **112,120** frontal-view chest X-rays from **30,805** unique patients
- **14 Disease Labels:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- **Split Strategy:** Patient-level 80/20 train-test split (prevents data leakage across patients)
- **Source:** [NIH ChestX-14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hybrid-cnn-transformer-chest-xray.git
cd hybrid-cnn-transformer-chest-xray
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```
Automatically downloads the NIH dataset via `kagglehub`, trains for 10 epochs, and saves `hybrid_model.pth`.

### 4. Launch the diagnostic dashboard
```bash
python app.py
```
Opens the Gradio interface at `http://localhost:7860` — upload a chest X-ray to get disease probabilities and the Grad-CAM heatmap.

---

## 📈 Evaluation

```python
from evaluate import evaluate_model, print_auc_scores

y_pred, y_true = evaluate_model(model, test_loader, device)
auc_scores, valid_labels = print_auc_scores(y_pred, y_true)
```

Evaluation uses **ROC-AUC** per disease class — the standard metric for multi-label medical image classification benchmarks.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| CNN Backbone | DenseNet-121 (torchvision, ImageNet pretrained) |
| Sequence Modelling | Transformer Encoder (PyTorch nn.Transformer) |
| Explainability | Grad-CAM + Transformer Attention Maps |
| Web Interface | Gradio |
| Data Processing | Pandas, NumPy, Pillow |
| Visualisation | Matplotlib, Seaborn, OpenCV |

---

## 👥 Team Members

| Name | Role |
|---|---|
| [Member 1] | Model Architecture & Training |
| [Member 2] | Dataset Preprocessing & Evaluation |
| [Member 3] | Grad-CAM & Explainability |
| [Member 4] | Dashboard & Gradio App |

---

## ⚠️ Disclaimer

This system is developed strictly for **academic and research purposes**. It is not a certified medical device and must not be used for clinical diagnosis or patient care.
