import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
import soundfile as sf

def read_librispeech(path, winsize=16000):
    files = list(Path(path).rglob('*.flac'))
    wavs = []
    for f in tqdm.tqdm(files[:10000]):
        finfo = sf.info(f)
        n_frames = finfo.frames
        wav_i = [sf.read(f,start=i,stop=i+winsize)[0] for i in range(0,n_frames-winsize,winsize)]
        wavs.extend(wav_i)
    wavs = np.array(wavs)

    return wavs[:-20], wavs[-20:]