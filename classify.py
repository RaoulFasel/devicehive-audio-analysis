import os
import numpy as np
from scipy.io import wavfile

from pydub import AudioSegment
from audio.processor import WavProcessor, format_predictions

sig = AudioSegment.from_file('train1.wav', format="wav")

start = int(0 * 1000)
end = int(100 * 1000)
segment = sig[start:end]
samples = np.array(segment.get_array_of_samples())
new_sig = samples.astype(np.float32)

with WavProcessor() as proc:
    predictions = proc.get_predictions(16000, new_sig)
    
    for p in predictions:
        print(p)