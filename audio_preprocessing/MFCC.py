# MFCC: Model trennt Anregungssignal und Impulsantwort des Filters
# Unterteilen in Fenster, Falten mit Fensterfunktion
# FFT, Betragsspektrum, Logarithmieren
# Zusammenfassen und Umrechnen mit Mel-Skala
# Dekorrelierte Werte durch (Rück-)Transformation / PCA

import librosa
import numpy as np
import matplotlib.pyplot as plt


def MFCC(audio_file: str):
    y, sr = librosa.load(audio_file, sr=None)
    y = librosa.util.normalize(y)

    n_fft = 1024
    hop_length = n_fft // 2
    n_mels = 128
    fft_n_amount = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


    log = librosa.power_to_db(fft_n_amount, ref=np.max)

    n_mfcc = 800
    mfcc = librosa.feature.mfcc(S=log, n_mfcc=n_mfcc)

    # Anzeige der MFCCs
    plt.figure(figsize=(10, 4))
    # Wie sollte der MFCC nun aussehen?

    librosa.display.specshow(log, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('Log')
    plt.show()
    
    # so?
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.show()

audio_file = 'audio_preprocessing/audios/talk_2.wav'
MFCC(audio_file)
