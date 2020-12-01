import glob
from sklearn.preprocessing import OneHotEncoder
import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm

def get(path):
    def frame(x,winsize,hopsize):
        frames = [x[i:i+winsize] for i in range(0,len(x)-winsize,hopsize)]
        return np.array(frames)

    frames = []
    for x in tqdm.tqdm(glob.glob(path)):
        audio,fs = librosa.core.load(x,sr=None)
        if len(audio)>33280:
            frames.append(frame(audio,winsize=33280,hopsize=33280))
    frames = np.concatenate(frames,axis=0)
    return frames[:-10], frames[-20:]