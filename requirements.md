# Tea Doctor - Requirements

## Python Version
- Python 3.8 or higher

## Required Packages

### Core Dependencies
```
streamlit>=1.28.0
tensorflow>=2.12.0
tensorflow-lite>=2.12.0
numpy>=1.21.0
opencv-python>=4.6.0
Pillow>=9.0.0
matplotlib>=3.5.0
```

### Installation Command
```bash
pip install -r requirements.txt
```

## System Requirements
- **Disk Space**: ~500 MB (for TensorFlow and dependencies)
- **RAM**: Minimum 2GB recommended
- **Processor**: Any modern CPU (GPU optional but not required)
- **OS**: Windows, macOS, or Linux

## Model File Required
- Place `tea_doctor_v7_final.tflite` in the same directory as the script
- Model Size: ~12 MB
- If missing: App runs in demo mode with simulated predictions

## Optional
- **Conda**: For isolated Python environments (recommended)
- **Virtual Environment**: Using `venv` (built-in with Python)
