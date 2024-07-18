import config as cfg
import numpy as np
import librosa

from python_speech_features import mfcc 
from helper import plot_spectrogram
from skimage import util
from typing import Tuple


def fft(segment: np.ndarray, rate: int = cfg.default_samplerate):
    segment_fft = np.fft.fft(segment)
    
    return segment_fft


def comp_mfcc(segment: np.ndarray, rate: int = cfg.default_samplerate):
    segment_mfcc = np.array(mfcc(segment, rate))

    return segment_mfcc


def countzerocrossings(segment: np.ndarray, rate: int = cfg.default_samplerate):
    counts = librosa.zero_crossings(segment)

    return counts


def averagevalue(segment: np.ndarray, rate: int = cfg.default_samplerate):
    avg = np.mean(np.abs(segment))

    return avg


def spectogram(segment: np.ndarray, rate: int = cfg.default_samplerate):
    spec = librosa.amplitude_to_db(np.abs(fft(segment)), ref=np.max)
    # plot_spectrogram(spec, 'full-pipeline-audio/spectrogram.png', rate=rate)

    return spec


def split_windows(data: np.ndarray, window_size: int = 1024, step_size: int = 1000):
    num_windows = (len(data) - window_size) // step_size + 1

    window = np.zeros((num_windows, window_size))

    for step in range(num_windows):
        start = step * step_size
        end = start + window_size
        if end > len(data):
            end = len(data)
        window[step] = data[start:end]

    return window


def split_data_into_windows(data: np.ndarray, window_size: int = cfg.default_windowsize,
                            step_size: int = int(0.5*cfg.default_windowsize)) -> np.ndarray:
    return util.view_as_windows(data.ravel(), window_shape=(window_size,), step=step_size)


# TODO: fix smth around here for the root:frame length.... error

def features(segment: np.ndarray, rate: int = cfg.default_samplerate):
    data_windows = split_data_into_windows(segment)
    
    for window in data_windows:
        spectogram(segment=window, rate=rate)
        fft(segment=window, rate=rate)
        comp_mfcc(segment=window, rate=rate)
        countzerocrossings(segment=window, rate=rate)
        averagevalue(segment=window, rate=rate)


