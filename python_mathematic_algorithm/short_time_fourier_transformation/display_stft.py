from fourier_transformation import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plot_spectrogram(audio_signal: np.ndarray, fs: int=1000, window: str='hamming',
                     nperseg: int=256, noverlap: int=128):
    
    nperseg = min(nperseg, len(audio_signal))
    # Sicherstellen, dass noverlap kleiner als nperseg ist
    noverlap = min(noverlap, nperseg - 1)
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


def plot_spectrogram_random(fs: int=1000, duration: float=5.0, freq: int=50, window: str='hamming',
                                  nperseg: int=256, noverlap: int=128):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    audio_signal = np.sin(2 * np.pi * freq * t)  
    # Füge Rauschen hinzu (optional)
    audio_signal += 0.0005 * np.random.randn(len(audio_signal))
    
    plot_spectrogram(audio_signal, fs, window, nperseg, noverlap)


if __name__ == '__main__':
    plot_spectrogram_random()

    randomdata = np.random.randint(-1, 1, 2000)     # Close call to "real" audio files 
    fft_data = np.abs(FastFourierTransformation_u_scipy(randomdata))
    normalize = fft_data / np.max(fft_data)

    plot_spectrogram(normalize)
