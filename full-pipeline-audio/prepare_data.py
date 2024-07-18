import librosa
import numpy as np
import python_speech_features
import os 


def getfiles(pathname: str):
    all_data = {}
    files = []

    path = "/".join(pathname.split("/"))

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)

    for f in files:
        if f.split(".")[-1] == "wav" or f.split(".")[-1] == "mp3":
            filename = os.path.join(path, f)
            data, samplerate = librosa.load(filename, sr=None)
            
            all_data[f] = data, samplerate

    return data, samplerate
