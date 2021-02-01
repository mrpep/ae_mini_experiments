from tensorflow.keras.callbacks import Callback
from datetime import datetime
from pathlib import Path
import pandas as pd
import copy
import wandb
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa

class WANDBLogger(Callback):
    def __init__(self, wandb_run = None, loggers=None):
        self.wandb_run = wandb_run
        self.step = 0
        self.epoch = 0
        self.loggers = loggers
        self.log_mapping = {'Spectrograms': self.log_spectrograms,
                            'TrainMetrics': self.log_train_metrics}
        
    def log_spectrograms(self, params, logs):
        inputs = [self.model.get_layer(l).output for l in params.get('in_layers',None)]
        outs = [self.model.get_layer(l).output for l in params.get('out_layers',None)]      

        predict_fn = tf.keras.backend.function(inputs=inputs,outputs=outs)
        plot_lims = params.get('plot_lims', [None, None])
        test_data = params.get('test_data',None)
        is_audio = params.get('is_audio', False)
        win_length = params.get('win_length',256)
        hop_length = params.get('hop_length',128)

        #if isinstance(test_data,BatchGenerator):
        #    x,y = test_data.__getitem__(0)
        #elif isinstance(test_data,list):
        x,y = test_data

        out_names = params.get('out_layers',None)
        y_pred = predict_fn(x)
        for i in range(len(y_pred[0])): #i->instancia j-> activacion
            sample_plots = []
            for j in range(len(y_pred)):
                out_name = out_names[j]
                plt.figure()
                title = '{}'.format(out_name.replace('/','-'))
                plt.title(title)
                if is_audio:
                    stft = librosa.core.stft(y_pred[j][i],hop_length=hop_length,win_length=win_length,n_fft=win_length)
                    plt.imshow(np.abs(stft),aspect='auto',origin='lower')
                else:
                    plt.imshow(np.squeeze(y_pred[j][i]).T,aspect='auto',origin='lower',vmin=plot_lims[0],vmax=plot_lims[1])
                sample_plots.append(wandb.Image(plt))
                plt.close()

            wandb.log({"sample_{}".format(i): sample_plots},step=self.step)

    def log_train_metrics(self, params, logs):
        prefix = params.get('prefix','')
        logs_ = {}
        for k,v in logs.items():
            if not isinstance(v,float):
                v = v.numpy()
            logs_['{}_{}'.format(prefix,k)] = v
        wandb.log(logs_,step=self.step)
        
    def on_epoch_end(self, batch, logs):
        for log_type, log_params in self.loggers.items():
            if (log_params['unit'] == 'epoch') and (self.epoch % int(log_params['freq']) == 0):
                self.log_mapping[log_type](log_params,logs)

        self.epoch += 1
        
    def on_batch_end(self, batch, logs):
        for log_type, log_params in self.loggers.items():
            if (log_params['unit'] == 'step') and (self.step % int(log_params['freq']) == 0):
                self.log_mapping[log_type](log_params, logs)

        self.step += 1