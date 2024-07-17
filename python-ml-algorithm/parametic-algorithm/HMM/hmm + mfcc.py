import numpy as np
import librosa
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm
from python_speech_features import mfcc


class AudioHMM:
    def __init__(self, num_states, num_mixtures):
        self.num_states = num_states
        self.num_mixtures = num_mixtures
        self.hmm_model = hmm.GMMHMM(n_components=num_states, n_mix=num_mixtures)
        
    def train(self, features):
        self.hmm_model.fit(features)
        
    def recognize(self, features):
        return self.hmm_model.predict(features)


def MFCC(audio_file: str):
    y, sr = librosa.load(audio_file, sr=None)
    y = librosa.util.normalize(y)

    n_fft = 1024
    hop_length = n_fft // 2
    n_mels = 128
    fft_n_amount = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


    log = librosa.power_to_db(fft_n_amount, ref=np.max)

    n_mfcc = 26
    mfcc = librosa.feature.mfcc(S=log, n_mfcc=n_mfcc)

    # Anzeige der MFCCs
    plt.figure(figsize=(10, 4))
    # Wie sollte der MFCC nun aussehen?
    # So?
    librosa.display.specshow(log, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('Specktogramm')
    plt.show()
    
    # Oder so?
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.show()

    return mfcc.T

audio_file = 'audio_preprocessing/audios/audio_to_prepare.mp3'
mfcc_features = MFCC(audio_file)
 
# Beispielkonfiguration
num_states = 15
num_mixtures = 9

audio_hmm = AudioHMM(num_states, num_mixtures)
audio_hmm.train(mfcc_features)


test_features = mfcc_features // 2  # 50 Frames von MFCCs für die Erkennung

predicted_states = audio_hmm.recognize(test_features)

print("Vorhergesagte versteckte Zustände:\n", predicted_states)
