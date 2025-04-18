import argparse
import os
import pdb
import sys
from timeit import default_timer as timer

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from tqdm import tqdm

from utilities import calculate_scalar

fs = 32000
nfft = 1024
hopsize = 320 # 640 for 20 ms
mel_bins = 96
window = 'hann'
fmin = 50
# hdf5_folder_name = '{}fs_{}nfft_{}hs_{}melb'.format(fs, nfft, hopsize, mel_bins)


class LogMelExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def transform(self, audio):
        # Ensure audio is a numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        # Check if the audio is mono or multi-channel
        if audio.ndim == 1:  # Mono audio
            # print(f"Processing mono audio: {audio.shape}")
            S = np.abs(librosa.stft(y=audio,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2

            S_mel = np.dot(self.melW, S).T
            S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
            return np.expand_dims(S_logmel, axis=0)  # Return with a single channel dimension

        elif audio.ndim == 2:  # Multi-channel audio
            # print(f"Processing multi-channel audio: {audio.shape}")
            channel_num = audio.shape[0]
            feature_logmel = []

            for n in range(channel_num):
                # print(f"Processing channel {n}: {audio[n].shape}")
                S = np.abs(librosa.stft(y=audio[n],
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize,
                                        center=True,
                                        window=self.window,
                                        pad_mode='reflect'))**2

                S_mel = np.dot(self.melW, S).T
                S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
                feature_logmel.append(S_mel)

            return np.stack(feature_logmel, axis=0)  # Stack features for all channels


def RT_preprocessing(audio, feature_type):

    extractor = LogMelExtractor(fs=fs, 
                                nfft=nfft,
                                hopsize=hopsize,
                                mel_bins=mel_bins,
                                window=window,
                                fmin=fmin)  

    # print("######################===================================================######################")
    # print(f"Audio shape: {audio.shape}")
    # print("######################===================================================######################")
    
    feature = extractor.transform(audio)
    '''(channels, seq_len, mel_bins)'''
    '''(channels, time, frequency)'''

    return feature

def extract_features(args):
    """
    Write features and infos of audios to hdf5.

    Args:
        dataset_dir: dataset path
        feature_dir: feature path
        data_type: 'dev' | 'eval'
        audio_type: 'foa' | 'mic'
    """

    # Path
    audio_dir = os.path.join(args.dataset_dir + '/wav/')
    meta_dir = os.path.join(args.dataset_dir + '/sets/'+ 'train_mos_list.txt')

    hdf5_dir = os.path.join(args.feature_dir)

    os.makedirs(os.path.dirname(hdf5_dir), exist_ok=True)

    begin_time = timer()
    audio_count = 0

    print('\n============> Start Extracting Features\n')
    
    iterator = tqdm(sorted(os.listdir(audio_dir)), total=len(os.listdir(audio_dir)), unit='it')

    for audio_fn in iterator:

        if audio_fn.endswith('.wav') and not audio_fn.startswith('.'):

            fn = audio_fn.split('.')[0]
            audio_path = os.path.join(audio_dir, audio_fn)
            # print(f"Audio path: {audio_path}")

            audio, _ = librosa.load(audio_path, sr=fs, mono=False, dtype=np.float32)
            '''(channel_nums, samples)'''
            audio_count += 1

            if np.sum(np.abs(audio)) < len(audio)*1e-4:
                with open("feature_removed.txt", "a+") as text_file:
                    print(f"Silent file removed in feature extractor: {audio_fn}", 
                        file=text_file)
                    tqdm.write("Silent file removed in feature extractor: {}".format(audio_fn))
                continue

            # features
            feature = RT_preprocessing(audio, args.feature_type)
            '''(channels, time, frequency)'''               

            # Read the CSV file
            df = pd.read_csv(os.path.join(meta_dir), sep=',', header=None)

            # Assign column names for clarity
            df.columns = ['filename', 'MOS1', 'MOS2']

            # Filter the row corresponding to the current file name
            row = df[df['filename'] == audio_fn]

            if row.empty:
                # print(f"Warning: No MOS values found for {audio_fn}")
                continue

            # Extract MOS1 and MOS2 values
            target_mos1 = row['MOS1'].values[0]  # Get the first value
            target_mos2 = row['MOS2'].values[0]  # Get the first value

            # print(target_mos1, target_mos2)

            hdf5_path = os.path.join(hdf5_dir, fn + '.h5')
            with h5py.File(hdf5_path, 'w') as hf:
                hf.create_dataset('feature', data=feature, dtype=np.float32)
                hf.create_group('target')
                hf['target'].create_dataset('MOS1', data=target_mos1, dtype=np.float32)
                hf['target'].create_dataset('MOS2', data=target_mos2, dtype=np.float32)

            tqdm.write('{}, {}, {}'.format(audio_count, hdf5_path, feature.shape))
    
    iterator.close()
    print("Extacting feature finished! Time spent: {:.3f} s".format(timer() - begin_time))


def fit(args):
    """
    Calculate scalar.

    Args:
        feature_dir: feature path
        data_type: 'dev' | 'eval'
        audio_type: 'foa' | 'mic'
    """
    
    hdf5_dir = os.path.join(args.feature_dir)

    scalar_path = os.path.join(args.feature_dir, 'scalar.h5')

    os.makedirs(os.path.dirname(scalar_path), exist_ok=True)

    print('\n============> Start Calculating Scalar.\n')

    load_time = timer()
    features = []
    for hdf5_fn in os.listdir(hdf5_dir):
        hdf5_path = os.path.join(hdf5_dir, hdf5_fn)
        with h5py.File(hdf5_path, 'r') as hf:
            features.append(hf['feature'][:])
    print('Load feature time: {:.3f} s'.format(timer() - load_time))

    features = np.concatenate(features, axis=1)
    (mean, std) = calculate_scalar(features)

    with h5py.File(scalar_path, 'w') as hf_scalar:
        hf_scalar.create_dataset('mean', data=mean, dtype=np.float32)
        hf_scalar.create_dataset('std', data=std, dtype=np.float32)

    print('Features shape: {}'.format(features.shape))
    print('mean {}:\n{}'.format(mean.shape, mean))
    print('std {}:\n{}'.format(std.shape, std))
    print('Write out scalar to {}'.format(scalar_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from audio file')

    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel'])                                

    args = parser.parse_args()

    extract_features(args)
    fit(args)
