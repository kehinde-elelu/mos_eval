import datetime
import itertools
import logging
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from librosa.display import specshow

from tqdm import tqdm

def calculate_scalar(features):

    mean = []
    std = []

    channels = features.shape[0]
    for channel in range(channels):
        feat = features[channel, :, :]
        mean.append(np.mean(feat, axis=0))
        std.append(np.std(feat, axis=0))

    mean = np.array(mean)
    std = np.array(std)
    mean = np.expand_dims(mean, axis=0)
    std = np.expand_dims(std, axis=0)
    mean = np.expand_dims(mean, axis=2)
    std = np.expand_dims(std, axis=2)

    return mean, std

