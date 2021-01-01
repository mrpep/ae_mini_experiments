import glob
from sklearn.preprocessing import OneHotEncoder
import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
from swissknife.aws import get_all_s3_objects, S3File
from swissknife.file import CompressedFile
import boto3

def get_audioset(n_zips, destination_path):
    bucket_name = 'lpepino-datasets2'
    prefix = 'audioset-wav-format'
    s3_client = boto3.client('s3')
    all_files = [k['Key'] for k in get_all_s3_objects(s3_client,Bucket=bucket_name,Prefix=prefix)]
    if not Path(destination_path).exists():
        Path(destination_path).mkdir(parents=True)
    for f in tqdm.tqdm(all_files[:n_zips]):
        destination_file = str(Path(destination_path,f.split('/')[-1]).expanduser().absolute())
        S3File('s3://{}/{}'.format(bucket_name,f)).download(destination_file)
        CompressedFile(destination_file).extract(destination_path)

def read_audioset(path, winsize=33280, hopsize=33280):
    def frame(x,winsize,hopsize):
        frames = [x[i:i+winsize] for i in range(0,len(x)-winsize,hopsize)]
        return np.array(frames)

    frames = []
    for x in tqdm.tqdm(glob.glob(path)):
        audio,fs = librosa.core.load(x,sr=None)
        if len(audio)>winsize:
            frames.append(frame(audio,winsize=winsize,hopsize=hopsize))
    frames = np.concatenate(frames,axis=0)
    return frames[:-10], frames[-20:]