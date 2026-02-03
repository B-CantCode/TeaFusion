#  Tea Doctor - Project Index & Quick Reference

##  Start Here

**Easiest Way (Windows):**
1. Double-click \
un.bat\
2. Wait ~15-30 seconds
3. App opens at http://localhost:8501

**Command Line (All Platforms):**

pip install -r requirements.txt
streamlit run tea_doctor_TFLITE_fixed.py


---

##  Project Files Explained

###  ESSENTIAL FILES

| File | Purpose | Size |
|------|---------|------|
| **tea_doctor_TFLITE_fixed.py** | Main app (Python) | 71 KB |
| **requirements.txt** | Python dependencies | <1 KB |
| **run.bat** | Windows launcher (one-click) | 1.7 KB |

###  DOCUMENTATION FILES

| File | Purpose | Read When |
|------|---------|-----------|
| **SETUP.md** | Installation guide | First time setup |
| **README.md** | Project overview | Want to know features |
| **requirements.md** | Technical requirements | Need package details |
| **PROJECT_COMPLETE.md** | Project summary | Quick overview |
| **THIS FILE** | Quick reference | Finding something fast |

---

##  Quick Commands

### Windows PowerShell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run tea_doctor_TFLITE_fixed.py

# Stop app
Ctrl+C

# Deactivate environment
deactivate


### macOS/Linux Terminal
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run tea_doctor_TFLITE_fixed.py

# Stop app
Ctrl+C

# Deactivate environment
deactivate


---

##  Accessing the App

### Local Machine
- URL: http://localhost:8501
- Access: Browser on same computer

### Same WiFi/Network
- URL: http://192.168.x.x:8501 (check terminal for exact IP)
- Access: Any device on same network

### Over Internet
- Not available in default setup
- Would require port forwarding or cloud deployment

---

##  Common Issues & Fixes

### Issue: "Python not found"

# Download Python from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH" during installation


### Issue: "No module named 'streamlit'"

# Make sure virtual environment is activated (check for (.venv) in prompt)
.\.venv\Scripts\Activate
pip install -r requirements.txt


### Issue: "Model file not found"
- App will run in DEMO mode with simulated predictions
- Place \	ea_doctor_v7_final.tflite\ in same folder for real predictions

### Issue: App starts but shows 'connection refused'

# Streamlit already running on same port?
# Kill existing process:
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force

# Try a different port:
streamlit run tea_doctor_TFLITE_fixed.py --server.port=8502


### Issue: "Out of memory" error
- Close other applications
- Restart app: \Ctrl+C\ then rerun
- First run takes more memory (TensorFlow initialization)

---

##  App Features Overview

### Upload & Analyze
1. Select language (English/Hindi/Assamese/Sanskrit)
2. Upload image OR use camera
3. See preprocessing
4. Get disease prediction + confidence
5. View treatment info in your language

### Main Features
-  91.2% accuracy
-  Kill Switch (bypass leaf checks)
-  Attention maps (see what model focused on)
-  Multilingual (4 languages)
-  Offline capable
-  Camera support
-  Demo mode (test without model)

### Disease Information
Each disease has:
- What it is
- How it spreads
- Causes
- Treatment options
- Prevention methods
- **All in your chosen language!**

---

##  Customization

### Change Default Language
Edit \	ea_doctor_TFLITE_fixed.py\, find:
\\\python
lang = st.sidebar.selectbox(..., ["en", "hi", "as", "sa"])
\\\
Change \"en"\ to desired language.

### Disable Demo Mode by Default
Find in code:
\\\python
demo_mode = st.sidebar.checkbox("Enable demo mode", value=False)
\\\
Already set to False (demo disabled by default).

### Adjust Confidence Threshold
Find in code:
\\\python
if confidence < 40:  # Change 40 to desired threshold
\\\

---

##  Support & Helplines

**Built-in Emergency Contacts:**
- Tea Board of India: 1800-345-3644
- Kisan Call Center: 1551
- Tocklai Tea Research: 0376-2360974
- Assam Agri University: 0376-2340001

Access in: About section  Emergency Helplines

---

##  Supported Languages

| Language | Code |
|----------|------|
| English | en |  
| हद | hi |  
| অসময | as |  
| सदर | sa |  

---

##  Disease Classes

1. **Brown_Blight** - Fungal, brown lesions
2. **Gray_Blight** - Fungal, rapid spread
3. **Healthy_leaf** - No disease 
4. **Helopeltis** - Mosquito bug pest
5. **Red_Rust** - Rust-like growth
6. **Red_spider** - Mite damage
7. **Sunlight_Scorching** - Heat damage (non-disease)

---

##  Performance

| Metric | Value |
|--------|-------|
| Accuracy | 91.2% |
| Model Size | 12 MB |
| Inference Time | <500ms |
| First Run Time | 10-15s (TF init) |
| Subsequent Runs | <1s |
| RAM Usage | ~500 MB |

---

##  Security & Privacy

-  **Offline**: No data sent to cloud
-  **Local**: Predictions computed locally
-  **Private**: Your images never leave your device
-  **No tracking**: No analytics or telemetry
-  **No accounts**: No login required

---

##  File Sizes & Requirements

`
Main Application:     71 KB (tea_doctor_TFLITE_fixed.py)
Model File:           12 MB (tea_doctor_v7_final.tflite) - optional
Dependencies:         ~300-400 MB (after pip install)
Total Disk Space:     ~500 MB needed
`

---



---

##  System Compatibility

 **Tested on:**
- Windows 10/11 (Python 3.10+)
- macOS 10.15+ (Python 3.8+)
- Ubuntu/Linux (Python 3.8+)

 **Browsers:**
- Chrome/Edge (best)
- Firefox
- Safari

---

---

##  You're Ready!

Everything is set up and documented. Choose your startup method:

**Windows Users:**
 Double-click \
un.bat\

**Mac/Linux Users:**
 Run: \streamlit run tea_doctor_TFLITE_fixed.py\

**Need Help?**
 Read \SETUP.md\

**Want Details?**
 Read \README.md\

---

**Last Updated**: February 1, 2026
**Project Status**:  COMPLETE & READY TO DEPLOY

