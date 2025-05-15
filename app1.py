import os
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import sounddevice as sd
from pydub import AudioSegment
import tempfile

# ... (kode import dan setup tetap)

# Load model
model = joblib.load('random_forest_model.pkl')

# Mapping ekspresi
expression_mapping = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "Surprised"
}

# Mapping lagu (pakai string key)
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
    "Surprised": [  # Catatan: tetap pakai huruf besar 'S' karena sesuai expression_mapping
        '"Eye of the Tiger" – Survivor',
        '"Take On Me" – a-ha',
        '"Don’t Stop Me Now" – Queen'
    ]
}

# ... (fungsi convert_to_wav & extract_features tetap sama)
# Konversi MP3 ke WAV jika perlu
def convert_to_wav(file_path, out_path='temp_converted.wav'):
    sound = AudioSegment.from_file(file_path)
    sound.export(out_path, format='wav')
    return out_path

# Ekstraksi fitur
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # RMSE
    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)

    # Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if pitch_values.size > 0 else 0.0

    # Gabungkan fitur
    features = np.hstack([mfcc_mean, pitch_mean, rmse_mean, zcr_mean])

    # Debug jumlah fitur
    print(f"Jumlah fitur: {len(features)}")  # Harus 16

    return features
# Streamlit setup
st.set_page_config(page_title="Deteksi Ekspresi Suara", layout="centered")
st.title("🎵 Deteksi Ekspresi Suara & Rekomendasi Lagu")

# Upload audio
uploaded_audio = st.file_uploader("Unggah file audio (wav/mp3)", type=["wav", "mp3"])

if uploaded_audio:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_audio.getbuffer())

    features = extract_features("temp_audio.wav")
    prediction = model.predict([features])
    label = prediction[0]

    # ⬇️ Taruh di sini
    expression = expression_mapping.get(label, "Unknown")
    songs = song_mapping.get(expression, [])

    st.success(f"Ekspresi terdeteksi: **{expression}**")
    st.write("🎧 Rekomendasi lagu: ")
    for song in songs:
        st.write(f"- {song}")

# Rekaman langsung
st.subheader("🎙️ Rekam Suara Langsung")

# Fungsi untuk merekam suara
def record_audio(duration=5, fs=16000):
    st.write("Mulai merekam...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Rekaman selesai.")
    return audio_data.flatten()

# Tombol untuk merekam suara
if st.button("Rekam Suara"):
    audio_data = record_audio()

    # Simpan rekaman dalam file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        sf.write(temp_file.name, audio_data, 16000)
        st.audio(temp_file.name)
        st.write("Proses prediksi sedang dilakukan...")

        features = extract_features(temp_file.name)
        prediction = model.predict([features])
        label = prediction[0]

        # ⬇️ Taruh juga di sini
        expression = expression_mapping.get(label, "Unknown")
        songs = song_mapping.get(expression, [])

        st.success(f"Ekspresi terdeteksi: **{expression}**")
        st.write("🎧 Rekomendasi lagu: ")
        for song in songs:
            st.write(f"- {song}")
