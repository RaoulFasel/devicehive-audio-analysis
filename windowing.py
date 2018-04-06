import os
import math
import operator
import json

from collections import Counter

import numpy as np
import pandas as pd

from scipy.io import wavfile
from pydub import AudioSegment
from audio.processor import WavProcessor, format_predictions

from log_config import LOGGING

proc = WavProcessor()

def get_labels(csv_file = 'labels.csv'): 
    df = pd.read_csv(csv_file)
    df = df.set_index('index')
    return df

def get_label_val(idx): 
    return [0.0 if val != 1 else 1.0 for val in df.loc[idx].values]

def label_sig(sig, df = get_labels()):
    global proc

    window_size = 6 * 1000
    overlap = 3 * 1000
    fragment_length = 900 * 1000

    current_length = 0

    window_start = 0
    window_end = window_start + window_size

    #Audio labels predicted by the algorithm for each window length.
    sig_labels = []

    while current_length < fragment_length:
        # print(window_start,window_end)

        segment = sig[window_start:window_end]
        samples = np.array(segment.get_array_of_samples())
        new_sig = samples.astype(np.float32)

        predictions = proc.get_predictions(16000, new_sig)

        for p in predictions:
            sig_labels.append(p)
            # print(str(window_start / (60*1000)), ":", str(window_end / (60*1000)), " - ", df.loc[p[0]].values[1], p[1])

        current_length = window_start
        window_start += (window_size - overlap)
        window_end = (window_start + window_size)
        
    return sig_labels

def classify_labels(labels, df = get_labels()):
    class_dict = {}

    total_items = 0

    for l in labels:
        total_items += 1

        if l[0] not in class_dict:
            class_dict[l[0]] = []

        class_dict[l[0]].append(l[1])

    class_dict_harmonic = {}

    for key, val in class_dict.items():
        class_prob = len(val)/float(total_items)
        class_prob = 1 / (class_prob)
        confidence_prob = 1 / np.mean(val)
        harmonic = 2 / ( class_prob + confidence_prob )
        class_dict_harmonic[key] = harmonic

    sorted_harmonics = sorted(class_dict_harmonic.items(), key=lambda x: x[1])

    classification_harmonics = {}

    df_labels = list(df)[2:]

    df_without_nan = df.dropna(thresh=3)
    total_labels = df_without_nan.shape[0]

    for label in df_labels:
        prob = []
        for key, val in sorted_harmonics:
            if not (math.isnan(df.loc[key][label])):
                prob.append(val)
            else:
                prob.append(0.0)
        harmonic = 2 / (1 / (df_without_nan[label].count()/total_labels) + 1 / np.mean(prob))
        classification_harmonics[label] = harmonic

    classification_harmonics_sorted = sorted(classification_harmonics.items(), key=lambda x: x[1])

    # for key, val in classification_harmonics_sorted:
    #     print(key,val)
        
    return classification_harmonics_sorted

def load_wav_from_folder(wav_folder = "wav"):
    signals = []

    for file in os.listdir(wav_folder):
        if file.endswith(".wav"):
            path = os.path.join(wav_folder, file)

            print('Started with ' + str(file))

            sig = load_wav(path)
            labels = label_sig(sig)
            res = classify_labels(labels)
            
            with open('result/' + str(file) + '_6_3_900.json', 'w') as f:
                json.dump(res, f)
                
            print('Saved ' + str(file) + '_6_3_900.json')
            print('---')
            
    return signals

def load_wav(path):
    sig = AudioSegment.from_file(path, format="wav")
    return sig

load_wav_from_folder()