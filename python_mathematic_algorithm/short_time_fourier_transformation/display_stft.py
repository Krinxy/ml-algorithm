from fourier_transformation import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_spectrogram_random_audio(audio_signal:np.ndarray = 0, fs: int=1000, duration: float=5.0,
                                  freq:int=50, window:str='hamming', nperseg:int=256, noverlap:int=128):

    if audio_signal == 0:
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        audio_signal_ex = np.sin(2 * np.pi * freq * t)  
        # Füge Rauschen hinzu (optional)
        # audio_signal_ex += 0.0005 * np.random.randn(len(audio_signal_ex))
        # Berechne die STFT
        frequencies, times, spectrogram_data = spectrogram(audio_signal_ex, fs=fs, window=window,
                                                            nperseg=nperseg, noverlap=noverlap)
    else:
        frequencies, times, spectrogram_data = spectrogram(audio_signal, fs=fs, window=window,
                                                           nperseg=nperseg, noverlap=noverlap)
    # Plot des Spektrogramms
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data))  
    plt.title('Spektrogramm')
    plt.ylabel('Frequenz (Hz)')
    plt.xlabel('Zeit (s)')
    plt.colorbar(label='Leistung (dB)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_spectrogram_random_audio()

    # TODO: Implement randomdata fft into plotting
 
    randomdata = np.random.randint(0, 100, 100)
    fft_data = FastFourierTransformation(randomdata)
    # plot_spectrogram_random_audio(audio_signal=fft_data)
