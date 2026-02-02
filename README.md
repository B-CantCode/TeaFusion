# Tea Doctor — Tea leaf disease detection (Streamlit + TFLite)

Lightweight Streamlit app to detect common tea leaf diseases using a TensorFlow Lite model.

This repository contains the UI, preprocessing, and inference helpers to run an offline-capable disease detector for tea leaves. The model binary (`.tflite`) is not included by default — see **Model** below for safe options.

## Features
- Preprocessing: denoising and lighting correction
- Structure-aware leaf checks (color + vein/edge analysis)
- TFLite inference with multi-input support (RGB, color features, texture features)
- Attention map visualization (heatmap overlay)
- Multilingual UI (English, Hindi, Assamese, Sanskrit fallback)

## Quick Start (Windows, PowerShell)
Prerequisites: Python 3.8+ and Git.

1. Clone the repo (or upload files via GitHub web UI):

```powershell
git clone https://github.com/B-CantCode/TeaFusion.git
cd "TeaFusion"
```

2. Create and activate a virtual environment (recommended):

```powershell
# create venv
python -m venv .venv
# activate
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the app:

```powershell
# recommended (works if streamlit is installed in the venv)
python -m streamlit run tea_doctor_TFLITE_fixed.py

# or use the provided launcher
.\run.bat
```

## Model (important)
- The TFLite model `tea_doctor_v7_final.tflite` is not included in the repository by default to avoid committing large binaries.
- Place the model file in the repository root so the app can find it, or use one of these recommended options:

1. GitHub Release: Upload the `.tflite` file as a GitHub Release asset and link to it in the README.
2. External storage: Host the model on cloud storage (S3, GDrive) and add a download script or link in `SETUP.md`.
3. Git LFS (if you want the model in the repo): enable Git LFS locally and track `*.tflite` (see below).

If you choose Git LFS, run locally before committing the model:

```powershell
git lfs install
git lfs track "*.tflite"
git add .gitattributes
git add tea_doctor_v7_final.tflite
git commit -m "Add model via Git LFS"
git push origin main
```

## Troubleshooting
- If `streamlit` is not found, ensure your venv is activated and `streamlit` is installed in it.
- If model loading fails, the app falls back to demo mode — see console messages for details.
- For plotting/Unicode issues on some OSes, ensure fonts like `DejaVu Sans` or `Arial Unicode MS` are available.

## Development notes
- Main app: `tea_doctor_TFLITE_fixed.py`
- Helper scripts and docs: `run.bat`, `SETUP.md`, `requirements.txt`
- Keep `*.tflite` out of git history unless tracked with LFS.

## Contributing
PRs welcome — please open an issue first if you plan substantial changes.

## License & Contact
Add your preferred license file (e.g., `LICENSE`) and contact info here.

---

If you want, I can also:
- Create a GitHub Release and upload a model asset for you (you'd provide the model file or allow upload), or
- Add a short README section with exact download links where you'd host the model.
# Tea Doctor — Streamlit TFLite App

This repository contains a Streamlit app `tea_doctor_TFLITE.py` for tea leaf disease detection.

Quick setup (Windows PowerShell):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install runtime dependencies (may include heavy packages like TensorFlow):

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install developer tools (optional):

```powershell
pip install -r dev-requirements.txt
```

4. Run the app

```powershell
streamlit run 'c:\\Users\\Admin\\Downloads\\tea_doctor_TFLITE.py'
```

Notes
- If you do not want TensorFlow installed (it is large), you can still run formatting and demo-mode features without it. To run predictions with a real model, place `tea_fusion_model.tflite` in the same folder and install TensorFlow.
- Use the VS Code tasks in `.vscode/tasks.json` to format (`Format file (black)`) and run (`Run Streamlit`).
