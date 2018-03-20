# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import argparse
import logging.config
import threading
import time
import os
import numpy as np
from scipy.io import wavfile
from log_config import LOGGING

from audio.captor import Captor
from audio.processor import WavProcessor, format_predictions


parser = argparse.ArgumentParser(description='Capture and process audio')
parser.add_argument('--min_time', type=float, default=5, metavar='SECONDS',
                    help='Minimum capture time')
parser.add_argument('--max_time', type=float, default=7, metavar='SECONDS',
                    help='Maximum capture time')
parser.add_argument('-s', '--save_path', type=str, metavar='PATH',
                    help='Save captured audio samples to provided path',
                    dest='path')

df = pd.read_csv('labels.csv')
df = df.set_index('index')
df = df[['Living room', 'Work place(kantoor)', 'Classroom', 'Work Shop', 'Kitchen', 'Bathroom', 'Bedroom', 'Trainstation', 'Airfield', 'On water', 'Public place', 'Roadside', 'Concert', 'Farm', 'Nature', 'Park']]

labels = ['Living room', 'Work place(kantoor)', 'Classroom', 'Work Shop', 'Kitchen', 'Bathroom', 'Bedroom', 'Trainstation', 'Airfield', 'On water', 'Public place', 'Roadside', 'Concert', 'Farm', 'Nature', 'Park']

def get_labels(idx): 
    return [0.0 if val != 1 else 1.0 for val in df.loc[idx].values]


logging.config.dictConfig(LOGGING)
logger = logging.getLogger('audio_analysis.capture')


class Capture(object):
    _ask_data = None
    _captor = None
    _save_path = None
    _processor_sleep_time = 0.01
    _process_buf = None
    _sample_rate = 16000

    def __init__(self, min_time, max_time, path=None):
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError('"{}" doesn\'t exist'.format(path))
            if not os.path.isdir(path):
                raise FileNotFoundError('"{}" isn\'t a directory'.format(path))

        self._save_path = path
        self._ask_data = threading.Event()
        self._captor = Captor(min_time, max_time, self._ask_data, self._process)

    def start(self):
        self._captor.start()
        self._process_loop()

    def _process(self, data):
        self._process_buf = np.frombuffer(data, dtype=np.int16)

    def _process_loop(self):
        with WavProcessor() as proc:
            self._ask_data.set()
            while True:
                i = 0
                pred_values = [0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]
                while i < 50:
                    if self._process_buf is None:
                        # Waiting for data to process
                        time.sleep(self._processor_sleep_time)
                        continue

                    self._ask_data.clear()
                    if self._save_path:
                        f_path = os.path.join(
                            self._save_path, 'record_{:.0f}.wav'.format(time.time())
                        )
                        wavfile.write(f_path, self._sample_rate, self._process_buf)
                        logger.info('"{}" saved.'.format(f_path))

                    
                    logger.info(str(i) +' Start processing.')
                    predictions = proc.get_predictions(
                        self._sample_rate, self._process_buf)
                    for p in predictions:
                        pred_values = [x + y for x, y in zip(pred_values, get_labels(p[0]))]
                        # print(get_labels(x[0]))
    #                logger.info(
    #                    'Predictions: {}'.format(format_predictions(predictions))
    #                )

                    logger.info('Stop processing.')
                    self._process_buf = None
                    self._ask_data.set()
                    i+=1


                #arr_index = np.argmax(np.array(pred_values))
                arr_index = np.argwhere(pred_values == np.amax(pred_values))
                arr_index = arr_index.flatten().tolist()
                #print(pred_values[int(arr_index)])
                #print(arr_index)
                # if(isinstance(arr_index,np.int64)):
                #     print(labels[int(arr_index)])
                # else:
                for x in arr_index:
                    print(labels[int(x)])
                
                print(pred_values)


if __name__ == '__main__':
    args = parser.parse_args()
    c = Capture(**vars(args))
    c.start()
