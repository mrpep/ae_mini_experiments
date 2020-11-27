import glob
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from sklearn.preprocessing import OneHotEncoder
import librosa
import os
import numpy as np
import tqdm
import wandb

from layers import *

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

all_wavs = glob.glob('../speechcommands/*/*.wav')
df_dict = [{'filename': k,'label': Path(k).parts[-2]} for k in all_wavs]
df_data = pd.DataFrame(df_dict)

commands_data = df_data[df_data['label'] != '_background_noise_']
ohe = OneHotEncoder()
labels = commands_data['label']
labels_oh = ohe.fit_transform(labels.values.reshape(-1,1)).todense()
filenames = commands_data.filename.values

rate = 16000
max_len = 16896

def read_file(x: tf.Tensor):
  wav, fs = librosa.core.load(x.numpy(),sr=rate)
  return wav

def pad_sequences(x):
  if x.shape[0]>max_len:
    x = x[:max_len]
  elif x.shape[0]<max_len:
    x = np.pad(x,((0,max_len - x.shape[0])))

  return x

batch_size = 64

def ae_1(max_len):
  x = tfkl.Input((max_len,),name='input_signal')
  pre = Spectrogram(win_size=1024,hop_size=512,fft_size=1024)(x)
  pre = MelScale(num_mel_bins=64,num_spectrogram_bins=513,sample_rate=16000,lower_edge_hertz=125,upper_edge_hertz=8000,name='original_spec')(pre)
  enc = tf.expand_dims(pre,axis=-1)
  enc = tfkl.Conv2D(64,6,strides=2,padding='SAME')(enc)
  enc = tfkl.BatchNormalization()(enc)
  enc = tfkl.LeakyReLU()(enc)
  enc = tfkl.Conv2D(128,6,strides=2,padding='SAME')(enc)
  enc = tfkl.BatchNormalization()(enc)
  enc = tfkl.LeakyReLU()(enc)
  enc = tfkl.Conv2D(256,6,strides=2,padding='SAME')(enc)
  enc = tfkl.BatchNormalization()(enc)
  enc = tfkl.LeakyReLU()(enc)
  enc = tfkl.Conv2D(512,6,strides=2,padding='SAME')(enc)
  enc = tfkl.BatchNormalization()(enc)
  enc = tfkl.LeakyReLU()(enc)
  enc = tfkl.Conv2D(512,6,strides=2,padding='SAME')(enc)
  dec = tfkl.Conv2DTranspose(512,6,strides=2,padding='SAME')(enc)
  dec = tfkl.BatchNormalization()(dec)
  dec = tfkl.Activation('relu')(dec)
  dec = tfkl.Conv2DTranspose(512,6,strides=2,padding='SAME')(dec)
  dec = tfkl.BatchNormalization()(dec)
  dec = tfkl.Activation('relu')(dec)
  dec = tfkl.Conv2DTranspose(256,6,strides=2,padding='SAME')(dec)
  dec = tfkl.BatchNormalization()(dec)
  dec = tfkl.Activation('relu')(dec)
  dec = tfkl.Conv2DTranspose(128,6,strides=2,padding='SAME')(dec)
  dec = tfkl.BatchNormalization()(dec)
  dec = tfkl.Activation('relu')(dec)
  dec = tfkl.Conv2DTranspose(64,6,strides=2,padding='SAME')(dec)
  dec = tfkl.BatchNormalization()(dec)
  dec = tfkl.Activation('relu')(dec)
  dec = tfkl.Conv2DTranspose(1,6,strides=1,padding='SAME')(dec)
  dec = Squeeze(axis=-1,name='spec_out')(dec)

  mse = (dec - pre)**2

  model = tf.keras.Model(inputs=x,outputs=mse)
  return model

model = ae_1(16896)
model.summary()

def mean_loss(y_true,y_pred):
  return tf.reduce_mean(y_pred)

model.compile(optimizer='adam',loss=mean_loss)
audio_train_data = [pad_sequences(librosa.core.load(x,sr=None)[0]) for x in tqdm.tqdm(commands_data.filename[:10000])]
audio_train_data = np.array(audio_train_data)

audio_test_data = [pad_sequences(librosa.core.load(x,sr=None)[0]) for x in tqdm.tqdm(commands_data.filename[9990:10010])]
audio_test_data = np.array(audio_test_data)

from callbacks import WANDBLogger

loggers = {'Spectrograms': {'test_data': [audio_test_data,audio_test_data],
                            'in_layers': ['input_signal'],
                            'out_layers': ['original_spec', 'spec_out'],
                            'freq': 1,
                            'unit': 'epoch'},
           'TrainMetrics': {'freq': 1, 'unit': 'step'}
          }

wandb.init(name='ae_1', project='ae_mini_experiments',config=model.get_config())

cbks = [WANDBLogger(loggers=loggers),tf.keras.callbacks.ModelCheckpoint('ckpts')]

model.fit(audio_train_data,audio_train_data,epochs=50,batch_size=batch_size,callbacks = cbks)