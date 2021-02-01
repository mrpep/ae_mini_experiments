import glob
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import librosa
import os
import numpy as np
import tqdm
import wandb

from layers import *
from models import *
from losses import *

import dienen

gpu_config = {'device': 'auto', 'allow_growth': True}
gpu_device = 0
gpu_growth = True

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        if len(gpus) < gpu_device:
            raise Exception('There are only {} available GPUs and the {} was requested'.format(len(gpus),gpu_device))
        tf.config.experimental.set_visible_devices(gpus[gpu_device], 'GPU')
    except RuntimeError as e:
        print(e)
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, gpu_growth)
    except RuntimeError as e:
        print(e)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

batch_size = 24

dienen_model = dienen.Model('models/vqvae_wav2vec_speech.yaml')
model = dienen_model.build()
model_config = dienen_model.original_config
#model = ae_1(16896)
model.summary()

def mean_loss(y_true,y_pred):
  return tf.reduce_mean(y_pred)

model.compile(optimizer=tf.keras.optimizers.RMSprop(clipnorm=1.0),loss=SpectralLoss())

#import gsc
#audio_train_data, audio_test_data = gsc.get_gsc()

#import audioset
#audioset.get_audioset(10,'../Datasets/Audioset/test_data')
#audio_train_data, audio_test_data = audioset.read_audioset('../Datasets/Audioset/test_data/*.wav',winsize=16640,hopsize=16640)

#import nsynth
#audio_train_data, audio_test_data = nsynth.read_nsynth('../nsynth')

import librispeech
audio_train_data, audio_test_data = librispeech.read_librispeech('../LibriSpeech')

from callbacks import WANDBLogger

loggers = {'Spectrograms': {'test_data': [audio_test_data,audio_test_data],
                            'in_layers': ['input_signal'],
                            'out_layers': ['input_signal', 'estimated_audio'],
                            'freq': 1,
                            'unit': 'epoch',
                            'is_audio': True},
           'TrainMetrics': {'freq': 1, 'unit': 'step'}
          }

wandb.init(name='vqvae_librispeech_raw', project='vqvae_librispeech_raw',config=model.get_config())
cbks = [WANDBLogger(loggers=loggers),tf.keras.callbacks.ModelCheckpoint('../ckpts/weights.{epoch:02d}-{loss:.2f}.hdf5')]

model.fit(audio_train_data,audio_train_data,epochs=50,batch_size=batch_size,callbacks = cbks)