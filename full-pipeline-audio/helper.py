import numpy as np
import matplotlib.pyplot as plt
import config as cfg


def plot_audio(data: np.array, filename: str):
    # Uses matplotlib to save a file with the plot of the audio data
    # BONUS: There is a memory leak in here. I haven't found it yet. +20 bonus exercise points for a demonstrable solution
    plt.close()
    plt.plot(data)
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.savefig(filename)
    plt.close()


def plot_spectrogram(data: np.array, filename: str, rate: int = cfg.default_samplerate):
    # This function clears the current plot, computes and saves a spectrogram to file
    plt.close()
    Pxx, freqs, bins, im = plt.specgram(data, Fs=rate)
    plt.ylabel('Frequenz [Hz]')
    plt.xlabel('Zeit [s]')
    plt.savefig(filename)
    plt.close()


def plot_histogram(data: np.array, filename: str):
    plt.close()
    _ = plt.hist(data, bins=100)
    plt.title("Histogram with 100 bins")
    plt.savefig(filename)
    plt.close()
