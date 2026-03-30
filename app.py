from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import os
import ffmpeg
from tensorflow.keras import layers, models

app = Flask(__name__)

# ======================
# Model (Lazy Load)
# ======================
model = None

def build_model():
    model = models.Sequential([
        layers.Input(shape=(13, 40, 1)),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(10, activation='softmax')
    ])
    return model

def get_model():
    global model
    if model is None:
        model = build_model()
        model.load_weights("digit_model.h5")
    return model

# ======================
# Preprocessing
# ======================
def preprocess(file_path, max_pad_len=40):
    audio, sr = librosa.load(file_path, sr=None)

    # Resample to 8000 Hz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=10)

    # Normalize
    audio = librosa.util.normalize(audio)

    # Fix length (1 sec)
    if len(audio) > 8000:
        audio = audio[:8000]
    else:
        audio = np.pad(audio, (0, 8000 - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=8000,
        n_mfcc=13,
        n_fft=512
    )

    mfcc = librosa.util.fix_length(mfcc, size=max_pad_len, axis=1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    return mfcc

# ======================
# Routes
# ======================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filename = file.filename

    # Save input file
    if filename.endswith(".webm"):
        input_path = "input.webm"
    else:
        input_path = "input.wav"

    file.save(input_path)

    output_path = "converted.wav"

    try:
        # 🔥 Convert WebM → WAV
        if input_path.endswith(".webm"):
            ffmpeg.input(input_path).output(
                output_path,
                ar=8000,
                ac=1
            ).run(overwrite_output=True)
        else:
            output_path = input_path

        # Preprocess
        features = preprocess(output_path)

        model = get_model()
        prediction = model.predict(features)
        digit = int(np.argmax(prediction))

        return jsonify({"predicted_digit": digit})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        for f in [input_path, output_path]:
            if os.path.exists(f):
                os.remove(f)
# ======================
# Run
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)