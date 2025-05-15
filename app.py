import os
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import sounddevice as sd
from pydub import AudioSegment
import tempfile

# Load model, scaler, dan encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Konversi MP3 ke WAV
def convert_to_wav(file_path, out_path='temp_converted.wav'):
    sound = AudioSegment.from_file(file_path)
    sound.export(out_path, format='wav')
    return out_path

# Ekstraksi fitur audio
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=3)  # Pastikan 3 detik

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # RMSE
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if pitch_values.size > 0 else 0.0

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        features = np.hstack([
            mfcc_mean,
            pitch_mean,
            rmse_mean,
            zcr_mean,
            chroma_mean,
            contrast_mean
        ])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prediksi & rekomendasi
def predict_and_recommend(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    label = prediction[0]
    expression = label_encoder.inverse_transform([label])[0]
    songs = song_mapping.get(expression.lower(), [])
    return expression, songs

# Rekomendasi lagu berdasarkan emosi
song_mapping = {
    "neutral": [
        '"Clocks" – Coldplay',
        '"Photograph" – Ed Sheeran',
        '"Viva La Vida" – Coldplay'
    ],
    "calm": [
        '"Weightless" – Marconi Union',
        '"Strawberry Swing" – Coldplay',
        '"Bloom" – The Paper Kites'
    ],
    "happy": [
        '"Happy" – Pharrell Williams',
        '"Can’t Stop the Feeling!" – Justin Timberlake',
        '"Good as Hell" – Lizzo'
    ],
    "sad": [
        '"Someone Like You" – Adele',
        '"The Night We Met" – Lord Huron',
        '"Hurt" – Johnny Cash'
    ],
    "angry": [
        '"Killing in the Name" – Rage Against the Machine',
        '"Break Stuff" – Limp Bizkit',
        '"Headstrong" – Trapt'
    ],
    "fearful": [
        '"The Sound of Silence" – Simon & Garfunkel',
        '"Breathe Me" – Sia',
        '"Mad World" – Gary Jules'
    ],
    "disgust": [
        '"Toxic" – Britney Spears',
        '"Bad Guy" – Billie Eilish',
        '"Uptown Funk" – Mark Ronson ft. Bruno Mars'
    ],
    "surprised": [
        '"Eye of the Tiger" – Survivor',
        '"Take On Me" – a-ha',
        '"Don’t Stop Me Now" – Queen'
    ]
}

# UI: Judul dan upload
st.set_page_config(page_title="Deteksi Ekspresi Suara", layout="centered")
st.title("🎵 Deteksi Ekspresi Suara & Rekomendasi Lagu")

# Upload audio file
uploaded_audio = st.file_uploader("Unggah file audio (wav/mp3)", type=["wav", "mp3"])
if uploaded_audio:
    ext = uploaded_audio.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_audio.getbuffer())
        tmp.flush()

        file_path = convert_to_wav(tmp.name) if ext == "mp3" else tmp.name
        features = extract_features(file_path)

        if features is not None:
            expression, songs = predict_and_recommend(features)
            st.success(f"Ekspresi terdeteksi: **{expression.upper()}**")
            st.write("🎧 Rekomendasi lagu:")
            for song in songs:
                st.write(f"- {song}")
        else:
            st.error("Gagal mengekstraksi fitur dari audio.")

# Rekaman langsung
st.subheader("🎙️ Rekam Suara Langsung")

def record_audio(duration=3, fs=16000):
    st.write("Mulai merekam...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Rekaman selesai.")
    return audio.flatten()

if st.button("Rekam Suara"):
    audio_data = record_audio()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        sf.write(temp_file.name, audio_data, 16000)
        st.audio(temp_file.name)

        features = extract_features(temp_file.name)
        if features is not None:
            expression, songs = predict_and_recommend(features)
            st.success(f"Ekspresi terdeteksi: **{expression.upper()}**")
            st.write("🎧 Rekomendasi lagu:")
            for song in songs:
                st.write(f"- {song}")
        else:
            st.error("Gagal mengekstraksi fitur dari rekaman.")
