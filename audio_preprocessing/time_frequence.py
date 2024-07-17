import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np 

audio_file1 = 'audio_preprocessing/audios/audio_to_prepare.mp3'
audio_file2 = 'audio_preprocessing/audios/screaming+song.mp3'
audio_file3 = 'audio_preprocessing/audios/talk_1.wav'
audio_file4 = 'audio_preprocessing/audios/talk_2.wav'

def Audio_Frequence(audio):
    y, sr = librosa.load(audio)

    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spektrogramm der Audio')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Frequenz (Hz)')
    plt.tight_layout()

    plt.savefig(fname='audio_preprocessing/spektrogramm', bbox_inches='tight')
    plt.show()

Audio_Frequence(audio_file1)
Audio_Frequence(audio_file2) 
Audio_Frequence(audio_file3)
Audio_Frequence(audio_file4)
