# Imports
import os
import sys

if "piper-sample-generator/" not in sys.path:
    sys.path.append("../piper-sample-generator/")
from generate_samples import generate_samples

import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm

output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    rir_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]}).cast_column("audio", datasets.Audio())
    # Save clips to 16-bit PCM wav files

    for row in tqdm(rir_dataset):
        name = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

if not os.path.exists("audioset"):
    os.mkdir("audioset")
    
    fname = "bal_train09.tar"
    out_dir = f"audioset/{fname}"
    link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
    #!wget -O out_dir link
    #cd audioset && tar -xvf bal_train09.tar

    output_dir = "./audioset_16k"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save clips to 16-bit PCM wav files
    audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]})
    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(audioset_dataset):
        name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))


# Free Music Archive dataset
# https://github.com/mdeff/fma

output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    # Save clips to 16-bit PCM wav files
    n_hours = 1  # use only 1 hour of clips for this example notebook, recommend increasing for full-scale training
    for i in tqdm(range(n_hours*3600//30)):  # this works because the FMA dataset is all 30 second clips
        row = next(fma_dataset)
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
        i += 1
        if i == n_hours*3600//30:
            break


# Download pre-computed openWakeWord features for training and validation

# training set (~2,000 hours from the ACAV100M Dataset)
# See https://huggingface.co/datasets/davidscripka/openwakeword_features for more information
#if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
    #!wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy

# validation set for false positive rate estimation (~11 hours)
#if not os.path.exists("validation_set_features.npy"):
    #!wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
