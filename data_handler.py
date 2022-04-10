import os
import shutil
import glob
import numpy as np
import pandas as pd
import pretty_midi
import pypianoroll
import tables
from music21 import converter, instrument, note, chord, stream
import music21
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.utils import np_utils
import json
import IPython.display
from datetime import datetime
import torch
import random
import itertools
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    def __init__(self, pianorolls, instrument_id):
        self.data = pianorolls
        self.length = int(pianorolls.size(1) / 32)
        self.instrument_id = instrument_id

    def __getitem__(self, index):
        sequence = self.data[self.instrument_id, (index * 32):((index+1) * 32), :]
        return sequence
      
    def __len__(self):
        return self.length

def load_data(root_dir, batch_size):
    data_dir = root_dir + '/lpd_5_cleansed'
    music_dataset_lpd_dir = root_dir + '/lmd_matched'

    combined_pianorolls = torch.load(os.path.join(root_dir, 'combined_1000_pianorolls.pt')) / 127.0
    pianoroll_lengths = torch.load(os.path.join(root_dir, 'combined_1000_pianorolls_lengths.pt'))
    pianoroll_lengths = pianoroll_lengths.numpy()
    pianoroll_cum_lengths = pianoroll_lengths.cumsum()

    pianorolls_list = []
    pianorolls_list.append(combined_pianorolls[:, :(pianoroll_cum_lengths[0] - pianoroll_cum_lengths[0] % 32), :])
    for i in range(len(pianoroll_cum_lengths) - 1):
        length = pianoroll_cum_lengths[i+1] - pianoroll_cum_lengths[i]
        # Get the nearest multiple of 32
        length_multiple = length - (length % 32)
        pianoroll = combined_pianorolls[:, pianoroll_cum_lengths[i]:(pianoroll_cum_lengths[i] + length_multiple), :]
        pianorolls_list.append(pianoroll)

    piano_dataset = CombinedDataset(combined_pianorolls, instrument_id=0)
    data_loader = DataLoader(piano_dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def load_gen_data(root_dir):
    data_dir = root_dir + '/lpd_5_cleansed'
    music_dataset_lpd_dir = root_dir + '/lmd_matched'

    combined_pianorolls = torch.load(os.path.join(root_dir, 'combined_1000_pianorolls.pt')) / 127.0
    pianoroll_lengths = torch.load(os.path.join(root_dir, 'combined_1000_pianorolls_lengths.pt'))
    pianoroll_lengths = pianoroll_lengths.numpy()
    pianoroll_cum_lengths = pianoroll_lengths.cumsum()

    pianorolls_list = []
    pianorolls_list.append(combined_pianorolls[:, :(pianoroll_cum_lengths[0] - pianoroll_cum_lengths[0] % 32), :])
    for i in range(len(pianoroll_cum_lengths) - 1):
        length = pianoroll_cum_lengths[i+1] - pianoroll_cum_lengths[i]
        # Get the nearest multiple of 32
        length_multiple = length - (length % 32)
        pianoroll = combined_pianorolls[:, pianoroll_cum_lengths[i]:(pianoroll_cum_lengths[i] + length_multiple), :]
        pianorolls_list.append(pianoroll)

    piano_dataset = CombinedDataset(combined_pianorolls, instrument_id=0)

    return piano_dataset
