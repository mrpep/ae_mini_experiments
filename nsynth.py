import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
import soundfile as sf

def read_nsynth(path, winsize=33024):
    files_train = list(Path(path, 'nsynth-train').rglob('*.wav'))
    files_val = list(Path(path, 'nsynth-valid').rglob('*.wav'))
    train_files = np.array([sf.read(f,start=0,stop=winsize)[0] for f in tqdm.tqdm(files_train[:30000])])
    val_files = np.array([sf.read(f,start=0,stop=winsize)[0] for f in tqdm.tqdm(files_val[:20])])

    return train_files, val_files