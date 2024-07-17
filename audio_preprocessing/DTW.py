from librosa.sequence import dtw
from pydub import AudioSegment

import librosa
import numpy as np


def DynamicTimeWrap(audio_file: str, segmentlength: int = 5000):

    audio = AudioSegment.from_file(audio_file)
    segment_length = segmentlength  # in millisec.
    segments = []

    for start_time in range(0, len(audio), segment_length):
        end_time = start_time + segment_length
        segment = audio[start_time:end_time]
        segment.export('temp_segment.wav', format='wav')        # Save(DTW)

        # Start DTW
        segment_abtastwerte, sr = librosa.load('temp_segment.wav')
        segment_mfcc = librosa.feature.mfcc(y=segment_abtastwerte, sr=sr)
        
        # DTW Distance
        ref_audio, sr = librosa.load(audio_file)
        ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr)
        D, wp = dtw(ref_mfcc, segment_mfcc, subseq=True, metric='euclidean')
        dtw_dist = np.min(D[:, -1])
        segments.append((segment, dtw_dist))
    
    segments.sort(key=lambda x: x[1])

    for segment, dtw_dist in segments:
        print(f"DTW-Distanz: {dtw_dist}")


audio_file = 'audio_preprocessing/audios/talk_2.wav'

DynamicTimeWrap(audio_file, 5000)
