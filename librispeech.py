import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
import soundfile as sf
from tensorflow.keras.utils import Sequence
from IPython import embed
from dsp import stft, get_default_window

def read_librispeech(path, winsize=16000, n_train=10000, n_test=20, calculate_spectrogram = False, frame_size = None,hop_size=None, window=None):
    files = list(Path(path).rglob('*.flac'))
    wavs = []
    if window is None:
        window = get_default_window(frame_size)[0]
    for f in tqdm.tqdm(files[:n_train + n_test]):
        finfo = sf.info(f)
        n_frames = finfo.frames
        wav_i = [sf.read(f,start=i,stop=i+winsize)[0] for i in range(0,n_frames-winsize,winsize)]
        if calculate_spectrogram:
            wav_i = [np.abs(stft(x_i,frame_size,hop_size,window)) for x_i in wav_i]
        wavs.extend(wav_i)
    wavs = np.array(wavs)

    return wavs[:n_train], wavs[-n_test:]

class LSGenerator(Sequence):
    def __init__(self,path, batch_size=16,winsize=16000,frame_size=256,hop_size=64):
        files = list(Path(path).rglob('*.flac'))
        dfs = []
        for f in tqdm.tqdm(files):
            finfo = sf.info(f)
            n_frames = finfo.frames
            starts = np.arange(0,n_frames-winsize,winsize)
            ends = starts + winsize
            filename = [str(f)]*len(starts)
            logids = [f.stem+'_{}'.format(i) for i in range(len(starts))]
            dfs.append(pd.DataFrame({'start': starts, 'end': ends, 'filename': filename, 'logid': logids}).set_index('logid'))
        self.data = pd.concat(dfs)
        self.winsize = winsize
        self.index = np.array(self.data.index)
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = get_default_window(self.frame_size)[0]

    def read_wav(self,x):
        try:
            return sf.read(x['filename'],start=x['start'],stop=x['end'])[0]
        except:
            return np.zeros((self.winsize,))

    def __getitem__(self,i):
        batch_idxs = np.take(self.index,np.arange(i*self.batch_size,(i+1)*self.batch_size),mode='wrap')
        batch_df = self.data.loc[batch_idxs]
        #batch_x = batch_df.apply(lambda x: sf.read(x['filename'],start=x['start'],stop=x['end'])[0],axis=1)
        batch_x = batch_df.apply(lambda x: self.read_wav(x),axis=1)
        batch_x = np.stack(batch_x)
        batch_x = np.array([stft(x_i,self.frame_size,self.hop_size,window=self.window) for x_i in batch_x])
        batch_x = np.abs(batch_x)

        return batch_x, batch_x

    def on_epoch_end(self):
        self.index = np.random.permutation(self.index)

    def __len__(self):
        return len(self.data)//self.batch_size

