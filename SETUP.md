#  Tea Doctor - Setup & Installation Guide

##  Quick Start (5 Minutes)

### Windows PowerShell
\\\powershell
# 1. Navigate to project folder
cd C:\Users\Admin\Downloads

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
.\.venv\Scripts\Activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the app
streamlit run tea_doctor_TFLITE_fixed.py
\\\

### macOS/Linux
\\\ash
# 1. Navigate to project folder
cd ~/tea_doctor_project

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate environment
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the app
streamlit run tea_doctor_TFLITE_fixed.py
\\\

##  System Requirements

### Minimum
- Python 3.8+
- 2GB RAM
- 500MB disk space
- Internet (for first-time installation only)

### Recommended
- Python 3.10+
- 4GB+ RAM
- 1GB disk space
- SSD for faster startup

##  Step-by-Step Installation

### 1. Verify Python Installation
\\\powershell
python --version  # Should show 3.8 or higher
\\\

### 2. Clone/Download Project
- Download the project folder
- Extract if zipped
- Navigate to the folder in terminal

### 3. Create Virtual Environment (Recommended)
\\\powershell
python -m venv .venv
\\\

**Why virtual environment?**
- Keeps dependencies isolated
- Prevents conflicts with other Python projects
- Makes uninstalling easy

### 4. Activate Virtual Environment

**Windows:**
\\\powershell
.\.venv\Scripts\Activate
\\\

**macOS/Linux:**
\\\ash
source .venv/bin/activate
\\\

**Success indicator:** Prompt will show \(.venv)\ prefix

### 5. Install Dependencies
\\\powershell
pip install --upgrade pip
pip install -r requirements.txt
\\\

**Troubleshooting install:**
- If slow: Add \-i https://pypi.org/simple/\ to pip command
- If errors: Run \pip install --upgrade pip setuptools wheel\ first

### 6. Add Model File
- Download or copy \	ea_doctor_v7_final.tflite\ to project folder
- Place it in the same directory as \	ea_doctor_TFLITE_fixed.py\
- File size should be ~12 MB

### 7. Run the Application
\\\powershell
streamlit run tea_doctor_TFLITE_fixed.py
\\\

**Expected output:**
\\\
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
\\\

##  Access the App

- **Local**: Open browser  http://localhost:8501
- **Network**: Use the Network URL from terminal (for other devices on same WiFi)
- **Default**: English language, Kill Switch disabled

##  Requirements File

\equirements.txt\ contains:
- streamlit (UI framework)
- tensorflow (ML framework)
- numpy (numerical computing)
- opencv-python (image processing)
- Pillow (image handling)
- matplotlib (visualization)

##  Troubleshooting

### App won't start
\\\powershell
# Clear cache and restart
Remove-Item -Recurse .streamlit/cache_*
streamlit run tea_doctor_TFLITE_fixed.py --logger.level=error
\\\

### ImportError: No module named 'streamlit'
\\\powershell
# Check virtual environment is activated
.\.venv\Scripts\Activate

# Reinstall
pip install -r requirements.txt
\\\

### Model file not found
- App will run in DEMO mode
- Predictions will be simulated with fixed seed
- To enable real predictions: place \	ea_doctor_v7_final.tflite\ in folder

### Slow on first run
- TensorFlow initializes: takes 10-15 seconds
- Subsequent runs are <500ms
- This is normal!

### Permission denied (macOS/Linux)
\\\ash
chmod +x tea_doctor_TFLITE_fixed.py
streamlit run tea_doctor_TFLITE_fixed.py
\\\

##  Verify Installation

\\\powershell
# 1. Check Python version
python --version

# 2. Check all packages installed
pip list | grep -E "streamlit|tensorflow|opencv"

# 3. Check model file exists
ls tea_doctor_v7_final.tflite

# 4. Verify Python syntax
python -m py_compile tea_doctor_TFLITE_fixed.py
\\\

##  Using the App

### First Time
1. Select language (English/Hindi/Assamese/Sanskrit)
2. Upload a tea leaf image or use camera
3. Wait for analysis (first run: ~15s, subsequent: ~1s)
4. View disease prediction and information

### Features
- **Kill Switch**: Skip leaf validation, rely on confidence
- **Demo Mode**: Test with simulated predictions
- **Attention Map**: See which parts influenced prediction
- **Multiple Languages**: All UI + disease info translated
- **Offline**: Works completely offline after setup

##  Getting Help

**If stuck:**
1. Check error message in terminal
2. Verify Python 3.8+ installed
3. Ensure virtual environment activated (check \(.venv)\ in prompt)
4. Reinstall: \pip install --force-reinstall -r requirements.txt\
5. Clear cache: \streamlit cache clear\

##  Next Steps

1.  Installation complete
2.  Run: \streamlit run tea_doctor_TFLITE_fixed.py\
3.  Upload a tea leaf image
4.  View disease prediction & info
5.  Try different languages in sidebar

---

**Happy disease detection! **
