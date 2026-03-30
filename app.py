from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
#from tensorflow.keras.models import load_model
import os
from tensorflow.keras import layers, models
app = Flask(__name__)

# ======================
# Load trained model
# ======================
#model = load_model("digit_model.keras",compile=False)
def build_model():
    model = models.Sequential([
        layers.Input(shape=(13,40,1)),

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
model = None

def get_model():
    global model
    if model is None:
        model = build_model()
        model.load_weights("digit_model.h5")
    return model
# ======================
# Preprocessing (MFCC)
# ======================
#def preprocess(file_path, max_pad_len=40):
#    audio, sr = librosa.load(file_path, sr=8000)

#    mfcc = librosa.feature.mfcc(
#        y=audio,
#        sr=sr,
#        n_mfcc=13,
#        n_fft=512
#    )

#    mfcc = librosa.util.fix_length(mfcc, size=max_pad_len, axis=1)

#    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1,13,40,1)

#    return mfcc

#def preprocess(file_path, max_pad_len=40):
#    audio, sr = librosa.load(file_path, sr=8000)

    # 🔥 NEW: remove silence
#    audio, _ = librosa.effects.trim(audio)

    # 🔥 NEW: normalize audio
#    audio = librosa.util.normalize(audio)

#    mfcc = librosa.feature.mfcc(
#        y=audio,
#        sr=sr,
#        n_mfcc=13,
#        n_fft=512
#    )

#    mfcc = librosa.util.fix_length(mfcc, size=max_pad_len, axis=1)

#    mfcc = mfcc[np.newaxis, ..., np.newaxis]

#    return mfcc
def preprocess(file_path, max_pad_len=40):
    audio, sr = librosa.load(file_path, sr=None)  # load original

    # 🔥 Resample to 8000 Hz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)

    # 🔥 Remove silence
    audio, _ = librosa.effects.trim(audio,top_db=20)

    # 🔥 Normalize
    audio = librosa.util.normalize(audio)

    # 🔥 Ensure fixed length (~1 sec)
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
    #return render_template("index.html")
    return "<h1>App is working</h1>"
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    temp_path = "temp.wav"
    file.save(temp_path)

    try:
        features = preprocess(temp_path)
        model = get_model()
        prediction = model.predict(features)
        digit = int(np.argmax(prediction))

        return jsonify({"predicted_digit": digit})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ======================
# Run app
# ======================
if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000,debug=False)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
