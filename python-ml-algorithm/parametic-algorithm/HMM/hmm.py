import numpy as np
from hmmlearn import hmm
from python_speech_features import mfcc

class AudioHMM:
    def __init__(self, num_states, num_mixtures):
        self.num_states = num_states
        self.num_mixtures = num_mixtures
        self.hmm_model = hmm.GMMHMM(n_components=num_states, n_mix=num_mixtures)
        
    def train(self, features):
        self.hmm_model.fit(features)
        
    def recognize(self, features):
        return self.hmm_model.predict(features)
    
# Beispielkonfiguration
num_states = 3
num_mixtures = 5

# Beispiel MFCCs (normalerweise von echten Audiodaten extrahiert)
mfcc_features = np.random.rand(100, 13)  # 100 Frames von MFCCs, jeweils mit 13 Koeffizienten

audio_hmm = AudioHMM(num_states, num_mixtures)
audio_hmm.train(mfcc_features)


test_features = np.random.rand(50, 13)  # 50 Frames von MFCCs für die Erkennung

predicted_states = audio_hmm.recognize(test_features)

print("Vorhergesagte versteckte Zustände:", predicted_states)
