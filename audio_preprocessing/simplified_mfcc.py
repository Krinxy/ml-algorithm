import librosa
import numpy as np
import matplotlib.pyplot as plt

# Lade dein Audiosignal und berechne die MFCCs
audio_file = 'audio_preprocessing/audios/talk_1.wav'
y, sr = librosa.load(audio_file, sr=None)
y = librosa.util.normalize(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Plot der MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.xlabel('Zeit (s)')
plt.ylabel('MFCC-Koeffizienten')
plt.tight_layout()
plt.show()
