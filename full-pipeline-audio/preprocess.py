import numpy as np
import librosa
import config as cfg
import copy

from helper import plot_audio, plot_histogram, plot_spectrogram

def convert_mono(audio: np.ndarray):
    if len(audio.shape) > 1:
        mono = audio[0]
    else:
        mono = audio

    return mono


def normalize(y: np.ndarray):
    y = librosa.util.normalize(y)

    return y


def adjustrange(data: np.ndarray):
    adjusted = data / np.max(np.abs(data), axis=0)

    return adjusted


def resample(data: np.ndarray, samplerate: int, target_samplerate: int = cfg.default_samplerate):
    ratio = target_samplerate / samplerate
    data_resampled = librosa.resample(data, orig_sr=samplerate, target_sr=target_samplerate)

    return data_resampled


def maximumfilter(data: np.ndarray, N: int = 21):
    copy_data = copy.deepcopy(data)
    filtered_audio = np.zeros_like(copy_data)

    for i in range(len(data)):
        start = max(0, i - N // 2)
        end = min(len(data), i + N // 2 + 1)
        window = data[start:end]

        maxvalue = np.max(window)

        filtered_audio[i] = maxvalue
    
    return filtered_audio


def meanfilter(data: np.ndarray, N: int = 15):
    copy_data = copy.deepcopy(data)

    filtered_audio = np.zeros_like(copy_data)

    for i in range(len(data)):
        start = max(0, i - N // 2)
        end = min(len(data), i + N // 2 + 1)
        window = data[start:end]

        meanvalue = np.mean(window)

        filtered_audio[i] = meanvalue

    return filtered_audio


def segment_words(data: np.ndarray, debug: bool = True):
    if debug:
        plot_audio(data=data, filename='full-pipeline-audio/data.png')
    
    max_data = maximumfilter(data, N=2001)

    if debug:
        plot_audio(data=max_data, filename='full-pipeline-audio/MaxN.png')
    
    smoothN = meanfilter(data=max_data)

    if debug:
        plot_audio(data=smoothN, filename='full-pipeline-audio/smoothN.png')
        plot_histogram(data=smoothN, filename='full-pipeline-audio/hist.png')

    words = []
    newword = np.array([])

    for i, sample in enumerate(smoothN):

        if sample > 0.005:
            newword = np.append(newword,data[i])
        if sample < 0.005 and len(newword) != 0:
            words.append(newword)
            newword = np.array([])
    
    return words
