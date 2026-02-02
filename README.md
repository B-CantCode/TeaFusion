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
