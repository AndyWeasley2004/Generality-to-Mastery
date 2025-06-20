import matplotlib.pyplot as plt
import os
from utils import pickle_load

dataset_root = '../score_data/full/pretrain_score'
category = os.listdir(dataset_root)
seq_lengths = 0

for each in category:
    path = os.path.join(dataset_root, each)
    if not os.path.isdir(path): continue
    
    for file in os.listdir(path):
        seq_lengths += len(pickle_load(os.path.join(path, file))[0]) # pickle[1] is bar indices, len of it is n_bars
    print(f'Category: {each}; Number of Bars in Total: {seq_lengths}')
    seq_lengths = 0
# plt.figure(figsize=(10, 8))
# plt.hist(seq_lengths, bins=20)
# plt.xlabel('Sequence Length', fontsize=14)
# plt.ylabel('File Counts', fontsize=14)
# plt.grid()
# plt.savefig('../performance_data/meta/maestro_full.png')