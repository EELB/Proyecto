import joblib
import numpy as np
from capture_and_extract import capture_audio, extract_features

# Cargar el modelo y el escalador
model = joblib.load('emotion_recognition_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_emotion(features):
    # Escalar características
    features = scaler.transform([features])
    prediction = model.predict(features)
    return prediction[0]

def main():
    # Capturar audio
    audio_data = capture_audio(duration=5)

    # Extraer características
    features = extract_features(audio_data)

    # Predecir la emoción
    emotion = predict_emotion(features)
    print(f"La emoción detectada es: {emotion}")

if __name__ == "__main__":
    main()
