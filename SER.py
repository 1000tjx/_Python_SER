import joblib
import numpy as np
import librosa
import soundfile
from sklearn.neural_network import MLPClassifier


class SER():
    def __init__(self, model_path='ser_model.joblib.compressed'):
        self.labels = ['sad', 'angry', 'happy']
        self.model = joblib.load(model_path)

    def extract_features(self, file_name):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32").flatten()
            sample_rate = sound_file.samplerate
            result = np.array([])
            # mfcc
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)  # noqa
            result = np.hstack((result, mfccs))
            # chroma
            # stft = np.abs(librosa.stft(X))
            # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # noqa
            # result = np.hstack((result, chroma))
            # mel
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)  # noqa
            result = np.hstack((result, mel))
            return result

    def predict(self, file_name):
        x = [self.extract_features(file_name), ]
        y_pred = self.model.predict(x)[0]
        return y_pred
