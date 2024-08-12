
# proyecto: Reconococimiento del habla mediante el habla
# Respnsable:Melissa Sanchez Martinez
# Version: 1.0
# Fecha de creaccion: 16 de julio 2024
# Fecha de modificacion: 22 de julio de 2024

import pyaudio
import numpy as np
import librosa

# Parámetros de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def capture_audio(duration=5):
    p = pyaudio.PyAudio()
    
    # Abrir stream de audio
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Capturando audio...")
    
    # Captura de audio
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
    
    # Detener y cerrar stream
    print("Detención de captura.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convertir a un array numpy
    audio_data = np.concatenate(frames)
    return audio_data

def extract_features(audio_data, sr=RATE):
    # Convertir a formato flotante
    audio_data = audio_data.astype(float)
    
    # Extraer características
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    
    # Promediar características en el tiempo
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1)
    ])
    return features
