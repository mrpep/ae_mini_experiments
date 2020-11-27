import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import copy

class Spectrogram(tfkl.Layer):
    def __init__(self,win_size,hop_size,fft_size=None,calculate='magnitude',window=tf.signal.hann_window,pad_end=False,name=None, trainable=False):
        super(Spectrogram, self).__init__(name=name)

        self.stft_args = {'ws': win_size,
                  'hs': hop_size,
                  'ffts': fft_size,
                  'win': window,
                  'pad': pad_end,
                  'calculate': calculate}

    def call(self,x):
        stft = tf.signal.stft(
                signals=x,
                frame_length=self.stft_args['ws'],
                frame_step=self.stft_args['hs'],
                fft_length=self.stft_args['ffts'],
                window_fn=self.stft_args['win'],
                pad_end=self.stft_args['pad'])

        calculate = self.stft_args['calculate']
        if calculate == 'magnitude':
            return tf.abs(stft)
        elif calculate == 'complex':
            return stft
        elif calculate == 'phase':
            return tf.math.angle(stft)
        else:
            raise Exception("{} not recognized as calculate parameter".format(calculate))

    def compute_output_shape(self,input_shape):
        signal_len = input_shape[-1]
        f_bins = self.stft_args['ws']//2 + 1
        t_bins = np.floor((signal_len-self.stft_args['ws']+self.stft_args['hs'])/self.stft_args['hs']).astype(int)
        output_shape = input_shape[:-1] + [t_bins,f_bins]

        return output_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'win_size': self.stft_args['ws'],
            'hop_size': self.stft_args['hs'],
            'fft_size': self.stft_args['ffts'],
            'calculate': self.stft_args['calculate'],
            'window': self.stft_args['win'],
            'pad_end': self.stft_args['pad']
        })
        return config

class MelScale(tfkl.Layer):
    def __init__(self,num_mel_bins=64,num_spectrogram_bins=None,sample_rate=None,lower_edge_hertz=125.0,upper_edge_hertz=3800.0,name=None, trainable=False): 
        super(MelScale, self).__init__(name=name)
        self.mel_args = {'mb':num_mel_bins, 
                         'sb':num_spectrogram_bins,
                         'sr': sample_rate,
                         'lh': lower_edge_hertz,
                         'uh': upper_edge_hertz}

    def call(self,x):
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.mel_args['mb'],
        num_spectrogram_bins=self.mel_args['sb'],
        sample_rate=self.mel_args['sr'],
        lower_edge_hertz=self.mel_args['lh'],
        upper_edge_hertz=self.mel_args['uh'])

        return tf.matmul(x,linear_to_mel_weight_matrix)

    def compute_output_shape(self,input_shape):
        num_mel_bins=self.mel_args['mb']
        output_shape = input_shape[:-1] + [num_mel_bins]
        return output_shape
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_mel_bins': self.mel_args['mb'],
            'num_spectrogram_bins': self.mel_args['sb'],
            'sample_rate': self.mel_args['sr'],
            'lower_edge_hertz': self.mel_args['lh'],
            'upper_edge_hertz': self.mel_args['uh']
        })
        return config

class Squeeze(tfkl.Layer):
    def __init__(self,axis=None, name = None, trainable=False):
        super(Squeeze, self).__init__(name=name)
        self.axis = axis
        
    def call(self, x):
        return tf.squeeze(x,axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config