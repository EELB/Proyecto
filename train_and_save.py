import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Función para extraer características de una muestra de audio
def extract_features(audio_data, sr=16000):
    audio_data = audio_data.astype(float)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1)
    ])
    return features

# Ejemplo de datos de entrenamiento (reemplaza con tus datos reales)
# Aquí se genera audio aleatorio para la demostración.
X = np.array([extract_features(np.random.randn(16000)) for _ in range(250)])
y = np.array(['happy' for _ in range(50)] + ['sad' for _ in range(50)]+ #entrenamos solamente 2 emociones
           #Agregregamos mas emociones entrenadas
             ['angry' for _ in range(50)] + 
             ['fear' for _ in range(50)] +
             ['depressed' for _ in range(50)])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar un modelo (ejemplo con SVC)
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Guardar el modelo y el escalador
joblib.dump(model, 'emotion_recognition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo y escalador guardados exitosamente.")
