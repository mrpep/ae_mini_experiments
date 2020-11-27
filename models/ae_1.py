from layers import *
import tensorflow as tf
import tensorflow.keras.layers as tfkl

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