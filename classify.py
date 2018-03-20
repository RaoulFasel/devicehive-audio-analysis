import os
import numpy as np
from scipy.io import wavfile
import pandas as pd

from pydub import AudioSegment
from audio.processor import WavProcessor, format_predictions

df = pd.read_csv('labels.csv')
df = df.set_index('index')

# print(df)

sig = AudioSegment.from_file('kitchen.wav', format="wav")

_start = 0
_end = 6

for x in range(0, 100):
    start = int(_start * 1000)
    end = int(_end * 1000)
    segment = sig[start:end]
    samples = np.array(segment.get_array_of_samples())
    new_sig = samples.astype(np.float32)

    with WavProcessor() as proc:
        predictions = proc.get_predictions(16000, new_sig)

        for p in predictions:
            print(str(_start/60), ":", str(_end/60), " - ", df.loc[p[0]].values[1], p[1])
    
    _start += 6
    _end += 6
    

