
import os
import scipy
import torch
import numpy as np
import logging
from pathlib import Path
import openwakeword
from openwakeword.data import generate_adversarial_texts, augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator

positive_train_output_dir = '../../piper_generated'
back_paths = ['../../audioset_16k', '../../fma']
rir_path = ['../../mit_rirs']

background_paths = []
for background_path in back_paths:
    background_paths.extend([i.path for i in os.scandir(background_path)])
rir_paths = [i.path for j in rir_path for i in os.scandir(j)]

n = 50  # sample size
positive_clips = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")]
duration_in_samples = []
for i in range(n):
    sr, dat = scipy.io.wavfile.read(positive_clips[np.random.randint(0, len(positive_clips))])
    duration_in_samples.append(len(dat))

total_length = int(round(np.median(duration_in_samples)/1000)*1000) + 12000  # add 750 ms to clip duration as buffer
print("is total_length 40000?", total_length)
import pdb; pdb.set_trace()
if total_length < 32000:
    total_length = 32000  # set a minimum of 32000 samples (2 seconds)
elif abs(total_length - 32000) <= 4000:
    total_length = 32000


positive_clips_train = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")]
positive_clips_train_generator = augment_clips(positive_clips_train, total_length=total_length,
                                                           batch_size=16,
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths)

logging.info("#"*50 + "\nComputing openwakeword features for generated samples\n" + "#"*50)
n_cpus = os.cpu_count()
if n_cpus is None:
    n_cpus = 1
else:
    n_cpus = n_cpus//2

compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),
                                            clip_duration=total_length,
                                            output_file="positive_pipers_train.npy",
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
