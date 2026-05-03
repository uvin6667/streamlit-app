import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairVision — Age Group Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(135deg, #2E4057, #3A6EA5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.05rem; color: #666; margin-top: 0.2rem;
    }
    .metric-card {
        background: #F2F6FA; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #3A6EA5;
        margin-bottom: 0.6rem;
    }
    .metric-label { font-size: 0.8rem; color: #888; font-weight: 600; }
    .metric-value { font-size: 1.5rem; font-weight: 800; color: #2E4057; }
    .top1-bar { background: #2E4057; }
    .top2-bar { background: #3A6EA5; }
    .top3-bar { background: #6FA8DC; }
    .warning-box {
        background: #FFF8E1; border: 1px solid #FFD54F;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #795548;
    }
    .info-box {
        background: #E8F4FD; border: 1px solid #90CAF9;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #1565C0;
    }
    .stProgress > div > div > div { background-color: #3A6EA5; }
    div[data-testid="stFileUploader"] { border: 2px dashed #3A6EA5; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Model definition (must match training code exactly) ───────────────────────
class FairVisionCNN(nn.Module):
    """
    Custom CNN for 9-class age group classification on FairFace.
    Architecture must exactly match the trained model.
    """
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "fairvision_demo_model.pt",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model = FairVisionCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ── Inference transform (must match eval_transform in training) ───────────────
def get_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Run prediction ─────────────────────────────────────────────────────────────
def predict(image: Image.Image, model, checkpoint):
    transform = get_transform(
        checkpoint["img_size"],
        checkpoint["mean"],
        checkpoint["std"],
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    age_names = checkpoint["age_names"]
    top3_idx  = probs.argsort()[::-1][:3]
    top3 = [(age_names[i], float(probs[i])) for i in top3_idx]
    all_probs = [(age_names[i], float(probs[i])) for i in range(len(age_names))]
    return top3, all_probs


# ── Confidence bar chart ───────────────────────────────────────────────────────
def make_bar_chart(all_probs):
    labels = [p[0] for p in all_probs]
    values = [p[1] * 100 for p in all_probs]
    max_idx = int(np.argmax(values))

    colors = ["#3A6EA5"] * len(labels)
    colors[max_idx] = "#2E4057"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9, color="#333")

    ax.set_xlim(0, max(values) * 1.25 + 5)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Prediction Confidence by Age Group", fontsize=11, fontweight="bold", color="#2E4057")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About FairVision")
    st.markdown("""
FairVision is a CNN-based age group classification system developed as part of the 
**Certified AI & ML Engineer (CAME)** programme at IJSE.

The model was trained on the [FairFace dataset](https://huggingface.co/datasets/HuggingFaceM4/FairFace)
using PyTorch and includes bias detection and mitigation analysis across race and gender groups.
    """)

    st.markdown("---")
    st.markdown("### 🏗️ Model Details")

    try:
        _, ckpt = load_model()
        st.markdown(f"""
| Property | Value |
|---|---|
| Architecture | FairVisionCNN |
| Parameters | ~2.4M |
| Input size | {ckpt['img_size']}×{ckpt['img_size']} px |
| Classes | {ckpt['num_classes']} age groups |
| Strategy | {ckpt['strategy']} |
| Test Accuracy | {ckpt['test_accuracy']:.2f}% |
        """)
    except Exception:
        st.info("Model not loaded yet.")

    st.markdown("---")
    st.markdown("### 📊 Fairness Summary")
    st.markdown("""
| Metric | Value |
|---|---|
| Race gap (baseline) | 11.16 pp |
| Race gap (M1 model) | 8.25 pp ↓ |
| Gender gap | 0.97 pp |
| 70+ F1 (baseline) | 0.017 |
| 70+ F1 (M1 model) | 0.438 ↑ |
    """)

    st.markdown("---")
    st.markdown("### ⚠️ Responsible Use")
    st.markdown("""
- For **demonstration purposes only**
- Do **not** use for consequential decisions
- Age estimates carry inherent uncertainty
- Performance varies across demographic groups
- Informed consent required for real use
    """)


# ── Main app ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">👁️ FairVision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN-Based Age Group Classification · Bias-Aware · FairFace Dataset · CAME Assignment</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model
try:
    model, checkpoint = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("""
    **Model file not found.**  
    Please place `fairvision_demo_model.pt` in the same directory as `app.py` and restart the app.
    """)
    model_loaded = False
    st.stop()

# Two column layout
col_upload, col_results = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown("### 📤 Upload a Face Image")
    st.markdown("""
Upload a clear, front-facing photograph. The model accepts any image format (JPG, PNG, WEBP).
For best results, use a well-lit image where the face is clearly visible and centred.
    """)

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Show image metadata
        w, h = image.size
        st.caption(f"Image size: {w}×{h} px  |  Resized to {checkpoint['img_size']}×{checkpoint['img_size']} for inference")


with col_results:
    st.markdown("### 🔍 Prediction Results")

    if not uploaded:
        st.markdown("""
<div class="info-box">
Upload a face image on the left to see the age group predictions here.
The model will display the <strong>top 3 predicted age groups</strong> with confidence scores.
</div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Running inference..."):
            top3, all_probs = predict(image, model, checkpoint)

        # ── Top prediction highlight ──────────────────────────────────────
        top_label, top_conf = top3[0]
        st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">TOP PREDICTION</div>
    <div class="metric-value">🎯 {top_label}</div>
    <div style="font-size:0.95rem; color:#3A6EA5; font-weight:600;">
        Confidence: {top_conf*100:.1f}%
    </div>
</div>
        """, unsafe_allow_html=True)

        # ── Top 3 predictions ─────────────────────────────────────────────
        st.markdown("#### Top 3 Predictions")
        medal = ["🥇", "🥈", "🥉"]
        bar_colors = ["#2E4057", "#3A6EA5", "#6FA8DC"]

        for rank, (label, conf) in enumerate(top3):
            cols = st.columns([0.08, 0.35, 0.57])
            with cols[0]:
                st.markdown(f"<div style='font-size:1.4rem;padding-top:0.3rem'>{medal[rank]}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div style='padding-top:0.55rem;font-weight:600;color:#2E4057'>{label}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.progress(conf, text=f"{conf*100:.1f}%")

        st.markdown("---")

        # ── Full probability chart ────────────────────────────────────────
        st.markdown("#### Full Confidence Distribution")
        fig = make_bar_chart(all_probs)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Limitations note ──────────────────────────────────────────────
        st.markdown("""
<div class="warning-box">
<strong>⚠️ Limitations:</strong> This is a research prototype trained on 64×64 images from the 
FairFace dataset. Accuracy is approximately 52–54% across 9 age classes. Performance may vary 
across demographic groups. Do not use this system to make consequential decisions about individuals.
</div>
        """, unsafe_allow_html=True)

# ── Bottom: How it works section ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 How It Works")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
**1. Dataset**  
Trained on [FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) — 
97,698 face images balanced across 7 race groups and 2 gender groups.
    """)
with c2:
    st.markdown("""
**2. Model**  
Custom 4-block CNN with Batch Normalisation and Dropout, trained from scratch 
in PyTorch for 25 epochs on a T4 GPU.
    """)
with c3:
    st.markdown("""
**3. Fairness**  
Audited across race and gender subgroups. Bias mitigation via smoothed 
class-weighted loss reduced the race accuracy gap from 11.2pp to 8.3pp.
    """)
with c4:
    st.markdown("""
**4. Limitations**  
~52–54% overall accuracy. Elderly (70+) and teenage (10–19) classes are 
hardest to classify. Full 224×224 training would improve results further.
    """)
