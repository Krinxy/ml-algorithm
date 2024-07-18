import time

from prepare_data import getfiles
from preprocess import *
from features import *


# Prepare Data
start_duration = time.time()
data, samplerate = getfiles('./audio_preprocessing/audios')
print('Samplerate: ', samplerate)
print(len(data))

# Preprocess
mono = convert_mono(data)
norm = normalize(mono)
adju = adjustrange(norm)
resa = resample(adju, samplerate)

# Segmentation
segments = segment_words(resa)

# Features
for segment in segments:
    features(segment)

print(segments)

end_duration = time.time()
print('Time Duration in seconds: ', round(end_duration - start_duration, 2))

