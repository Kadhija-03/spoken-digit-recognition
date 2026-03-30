## 📊 Dataset Used

This project uses the **Free Spoken Digit Dataset (FSDD)**.

🔗 Dataset Link: https://github.com/Jakobovski/free-spoken-digit-dataset

### 📌 Dataset Description
- Contains audio recordings of spoken digits (0–9)
- Total samples: ~3000 recordings
- Format: `.wav`
- Sampling rate: 8000 Hz
- Each audio file contains one spoken digit

### 📁 File Naming Format

 digit_speaker_index.wav
  Example: 5_theo_37.wav


### 📌 Dataset Characteristics
- Multiple speakers
- English spoken digits
- Clean and short audio clips

### 📌 Why this dataset?
- Suitable for speech recognition tasks
- Small and easy to train
- Good for real-time applications

---

## ⚙️ Preprocessing Steps
- Audio resampled to 8000 Hz
- Noise trimming
- Normalization
- MFCC feature extraction (13 coefficients)
- Padding to fixed length (40 frames)

---

## 🎯 Task
Classify spoken audio into digits (0–9)
