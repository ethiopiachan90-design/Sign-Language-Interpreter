# ✋ Sign Language to Text & Speech Conversion

Real-time American Sign Language (ASL) recognition system that translates hand gestures into text and spoken words using a webcam.

---

## 🚀 One-Click Setup (New PC)

> **Prerequisites:** Python 3.11 must be installed. [Download here](https://www.python.org/downloads/)  
> During installation, make sure to ✅ **"Add Python to PATH"**

### Steps:
1. **Copy `cnn8grps_rad1_model.h5`** into this folder (the model file — get it from the original USB/drive)
2. **Double-click `SETUP_AND_RUN.bat`**
3. That's it! The script will automatically:
   - Create a virtual environment
   - Install all dependencies
   - Download the MediaPipe hand tracking model
   - Launch the application

---

## 📁 Project Structure

```
Sign-Language-To-Text-and-Speech-Conversion/
│
├── SETUP_AND_RUN.bat         ← ONE-CLICK SETUP & RUN (start here)
├── final_pred.py             ← Main application (MediaPipe + CNN)
├── requirements.txt          ← Python dependencies
├── hand_landmarker.task      ← MediaPipe hand model (auto-downloaded)
├── cnn8grps_rad1_model.h5    ← CNN sign classifier (copy manually)
│
├── sign_language_dashboard.py  ← Alternative dashboard UI
├── sign_to_speech_ui.py        ← Alternative speech-focused UI
├── prediction_wo_gui.py        ← Terminal-only prediction mode
├── data_collection_final.py    ← Tool to collect training data
│
└── AtoZ_3.1/                   ← Training dataset (A-Z gesture images)
```

---

## 🛠 Manual Setup (if the bat file doesn't work)

```bash
# 1. Create virtual environment
python -m venv env

# 2. Activate it
env\Scripts\activate       # Windows
source env/bin/activate    # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python final_pred.py
```

---

## 📋 Requirements

| Package | Version |
|---|---|
| TensorFlow | 2.16.1 |
| Keras | 3.1.1 |
| MediaPipe | 0.10.11 |
| OpenCV | latest |
| pyttsx3 | latest |
| Pillow | latest |

> **Note:** `pyenchant` (word suggestions) may fail on some systems — the app still works without it.

---

## 🎯 How to Use

1. Launch the app via `SETUP_AND_RUN.bat`
2. Show your hand in the **webcam view** (right side of screen)
3. Hold each ASL letter gesture steady for ~0.5 seconds — it will be detected
4. Use the **"next"** gesture (flat hand facing forward) to commit a letter
5. Press **"Speak"** to hear the sentence read aloud
6. Press **"Clear"** to reset

---

## 📦 Model File Note

The CNN model (`cnn8grps_rad1_model.h5`) is **~13MB** — too large for GitHub's free tier.  
**Keep a copy on:**
- Your USB drive
- Google Drive / OneDrive
- Or re-train using `data_collection_final.py`

---

## 👥 Team G18
