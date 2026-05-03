# FairVision — Streamlit Demo App

CNN-based age group classification with bias detection and mitigation.  
Built for the CAME Individual Assignment 1 at IJSE (2025/2026).

---

## Files

```
fairvision_app/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── fairvision_demo_model.pt  ← Trained model (copy from Colab — see below)
└── README.md
```

---

## Step 1: Get the model file from Colab

After running all cells in the notebook, download `fairvision_demo_model.pt` from Colab:

```python
# Run this in a Colab cell to download the file
from google.colab import files
files.download('fairvision_demo_model.pt')
```

Place the downloaded file inside this `fairvision_app/` folder.

---

## Step 2: Deploy on Streamlit Cloud (free, public URL)

1. Push this entire `fairvision_app/` folder to a **GitHub repository**  
   (make sure `fairvision_demo_model.pt` is included)

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **"New app"** and fill in:
   - **Repository:** your GitHub repo
   - **Branch:** `main`
   - **Main file path:** `fairvision_app/app.py`  
     (or just `app.py` if you pushed the files to the repo root)

4. Click **"Deploy"** — Streamlit Cloud will install dependencies and launch the app

5. Copy the public URL (format: `https://your-name-fairvision-app-xyz.streamlit.app`)  
   → Paste this URL into Section 9 of your technical report

> ⚠️ Keep the app running until grading is complete. Streamlit Cloud free tier  
> hibernates after inactivity — just reopen the URL to wake it up.

---

## Run locally (optional, for testing)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Model details

| Property | Value |
|---|---|
| Architecture | FairVisionCNN (4 conv blocks, custom) |
| Parameters | ~2.4M |
| Input size | 64×64 px |
| Output | 9 age group classes |
| Strategy | M1: Smoothed Class-Weighted Loss |
| Test Accuracy | ~52.72% |
| Dataset | FairFace 0.25 config (HuggingFace) |
