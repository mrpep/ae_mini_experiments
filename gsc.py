import glob
from sklearn.preprocessing import OneHotEncoder
import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm

def get_gsc(path='../speechcommands/*/*.wav'):

    all_wavs = glob.glob(path)
    df_dict = [{'filename': k,'label': Path(k).parts[-2]} for k in all_wavs]
    df_data = pd.DataFrame(df_dict)

    commands_data = df_data[df_data['label'] != '_background_noise_']
    ohe = OneHotEncoder()
    labels = commands_data['label']
    labels_oh = ohe.fit_transform(labels.values.reshape(-1,1)).todense()
    filenames = commands_data.filename.values

    rate = 16000
    max_len = 16896

    def pad_sequences(x):
    if x.shape[0]>max_len:
        x = x[:max_len]
    elif x.shape[0]<max_len:
        x = np.pad(x,((0,max_len - x.shape[0])))

    train_data = [pad_sequences(librosa.core.load(x,sr=None)[0]) for x in tqdm.tqdm(commands_data.filename[:10000])]
    train_data = np.array(train_data)

    test_data = [pad_sequences(librosa.core.load(x,sr=None)[0]) for x in tqdm.tqdm(commands_data.filename[9990:10010])]
    test_data = np.array(test_data)

    return train_data, test_data