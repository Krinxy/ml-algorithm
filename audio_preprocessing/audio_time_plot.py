import librosa
import matplotlib.pyplot as plt
import numpy as np

audio_file1 = 'audio_preprocessing/audio_to_prepare.mp3'
audio_file2 = 'audio_preprocessing/screaming+song.mp3'
audio_file3 = 'audio_preprocessing/talk_1.wav'
audio_file4 = 'audio_preprocessing/talk_2.wav'

def audio_time_plot(audio_file:str):
    y, sr = librosa.load(audio_file)
    time = np.arange(0, len(y)) / sr


    plt.figure(figsize=(14, 5))
    plt.plot(time, y)
    plt.title('Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# For saving a plot, comment out everything you don't need

audio_time_plot(audio_file1)
audio_time_plot(audio_file2)
audio_time_plot(audio_file3)
audio_time_plot(audio_file4)
