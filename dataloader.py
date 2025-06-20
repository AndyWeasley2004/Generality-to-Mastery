import os
import random
from glob import glob

import torch
import numpy as np
from torch.utils.data import Dataset
from utils import pickle_load


def check_extreme_pitch_score(raw_events):
    low, high = 128, 0
    for ev in raw_events:
        if 'Note_Pitch' in ev or 'Note_On' in ev: # for score and performance representaions
            val = int(ev.split('_')[-1])
            low = min(low, val)
            high = max(high, val)
    return low, high


def transpose_events(raw_events, semitone_shift):
    transposed = []
    for ev in raw_events:
        if 'Note_On' in ev or 'Note_Off' in ev:
            _, prefix, pitch_str = ev.split('_')
            pitch = int(pitch_str)
            transposed.append(f'Note_{prefix}_{pitch + semitone_shift}')
        else:
            transposed.append(ev)
    return transposed

def transpose_score(raw_events, semitone_shift):
    transposed = []
    for ev in raw_events:
        if 'Note_Pitch' in ev:
            pitch_str = ev.split('_')[-1]
            pitch = int(pitch_str)
            transposed.append(f'Note_Pitch_{pitch + semitone_shift}')
        else:
            transposed.append(ev)
    return transposed

def convert_event(event_seq, event2idx, to_ndarr=True):
    seq = [event2idx[e] for e in event_seq]
    if to_ndarr:
        seq = np.array(seq, dtype=np.int16)
    return seq

class MusicSegmentDataset(Dataset):
    def __init__(self, data_dir, vocab_path, model_dec_seqlen=2400, 
                augment=False, from_start_only=False, data_type='score', 
                split='train', train_ratio=0.8, random_seed=42,
                used_categories=None, epoch_steps=None, eval=False):
        self.data_dir = data_dir
        self.vocab_file = vocab_path
        self.model_dec_seqlen = model_dec_seqlen
        self.augment = augment
        self.eval = eval
        self.from_start_only = from_start_only
        self.augment_range = range(-3, 4)
        self.data_type = data_type
        self.used_categories = used_categories if used_categories else os.listdir(data_dir)
        self.epoch_steps = epoch_steps

        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        self.read_vocab()
        self.build_filelist()

    def read_vocab(self):
        vocab_pack = pickle_load(self.vocab_file)
        self.event2idx = vocab_pack[0]
        self.idx2event = vocab_pack[1]
        self.pad_token = len(self.event2idx)
        self.event2idx['PAD_None'] = self.pad_token
        self.vocab_size = self.pad_token + 1

    def build_filelist(self):
        self.category_files = {}
        for category in sorted(os.listdir(self.data_dir)):
            if category not in self.used_categories: continue
            cat_dir = os.path.join(self.data_dir, category)
            if os.path.isdir(cat_dir):
                files = sorted(glob(os.path.join(cat_dir, '*.pkl')))
                
                if files:
                    rng = random.Random(self.random_seed)
                    rng.shuffle(files)
                    split_index = int(len(files) * self.train_ratio)
                    if self.split == 'train':
                        self.category_files[category] = files[:split_index]
                    elif self.split == 'valid':
                        self.category_files[category] = files[split_index:]
                    else:
                        self.category_files[category] = files
        self.categories = list(self.category_files.keys())

    def __len__(self):
        if self.split == 'train':
            if self.epoch_steps:
                return len(self.categories) * self.epoch_steps
            else:
                return len(self.categories) * 2500
        else:
            return sum(len(files) for files in self.category_files.values())

    def sample_segment(self, events_str, seg_indices):
        total_length = len(events_str)
        max_allowed_start_value = total_length - (self.model_dec_seqlen // 2 - 2)
        valid_start_indices = [i for i, pos in enumerate(seg_indices[:-1]) if pos <= max_allowed_start_value]

        if self.from_start_only or not valid_start_indices:
            start_idx = 0
        else:
            start_idx = random.choice(valid_start_indices)

        start = seg_indices[start_idx]
        max_end = None

        for candidate in seg_indices[start_idx + 1:]:
            if (candidate - start + 2) <= self.model_dec_seqlen:
                max_end = candidate
            else:
                break
        if max_end is None:
            max_end = total_length
        return start, max_end

    def __getitem__(self, idx):
        if self.split == 'train':
            category = random.choice(self.categories)
            file_path = random.choice(self.category_files[category])
        else:
            all_files = []
            for cat in self.categories:
                all_files.extend(self.category_files[cat])
            file_path = all_files[idx]

        actual_events, seg_indices = pickle_load(file_path)
        if len(seg_indices) == 0 or seg_indices[0] != 0:
            seg_indices.insert(0, 0)
        events_str = ['{}_{}'.format(x['name'], x['value']) for x in actual_events]

        composer_token = events_str[0]
        tempo_token = events_str[1]

        start, end = self.sample_segment(events_str, seg_indices)
        seg_tokens = events_str[start:end+1] if end != len(events_str)-1 else events_str[start:end]

        if self.augment:
            seg_tokens = self.pitch_augment(seg_tokens)

        if start != 0:
            if self.data_type == 'score' and not self.eval:
                seg_tokens = [composer_token, tempo_token] + seg_tokens
            elif self.data_type == 'score' and self.eval:
                seg_tokens = [tempo_token] + seg_tokens
            elif self.data_type == 'performance':
                seg_tokens = [composer_token] + seg_tokens

        token_ids = convert_event(seg_tokens, self.event2idx, to_ndarr=False)
        dec_inp = token_ids[:-1]
        dec_tgt = token_ids[1:]

        chord_mask = []
        melody_mask = []
        for tid in dec_tgt:
            name_part = self.idx2event[tid].split('_')[:-1]
            chord_mask.append(1 if 'Chord' in name_part else 0)
            melody_mask.append(1 if 'Note' in name_part else 0)

        segment_data = {
            'dec_inp': np.array(dec_inp, dtype=np.int16),
            'dec_tgt': np.array(dec_tgt, dtype=np.int16),
            'inp_chord': np.array(chord_mask, dtype=np.int8),
            'inp_melody': np.array(melody_mask, dtype=np.int8),
            'length': len(dec_inp),
            'filename': os.path.basename(file_path),
            'id': idx,
            'composer': composer_token
        }
        return segment_data
    
    def pitch_augment(self, events):
        low, high = check_extreme_pitch_score(events)
        pitch_shift = random.choice(self.augment_range)
        while low + pitch_shift < 21 or high + pitch_shift > 108:
            pitch_shift = random.choice(self.augment_range)
        events = transpose_score(events, pitch_shift)
        return events

    def collate_fn(self, batch):
        dec_inp_list = []
        dec_tgt_list = []
        chord_mask_list = []
        melody_mask_list = []
        length_list = []
        id_list = []
        filename_list = []
        composer_list = []

        for seg in batch:
            curr_len = seg['length']
            trunc_len = min(curr_len, self.model_dec_seqlen)
            tmp_inp = seg['dec_inp'][:trunc_len]
            tmp_inp = np.pad(tmp_inp, (0, self.model_dec_seqlen - trunc_len),
                        'constant', constant_values=self.pad_token)
            
            tmp_tgt = seg['dec_tgt'][:trunc_len]
            tmp_tgt = np.pad(tmp_tgt, (0, self.model_dec_seqlen - trunc_len),
                        'constant', constant_values=self.pad_token)
            
            tmp_chord = seg['inp_chord'][:trunc_len]
            tmp_chord = np.pad(tmp_chord, (0, self.model_dec_seqlen - trunc_len),
                        'constant', constant_values=self.pad_token)

            tmp_melody = seg['inp_melody'][:trunc_len]
            tmp_melody = np.pad(tmp_melody, (0, self.model_dec_seqlen - trunc_len),
                        'constant', constant_values=self.pad_token)
            
            dec_inp_list.append(tmp_inp)
            dec_tgt_list.append(tmp_tgt)
            chord_mask_list.append(tmp_chord)
            melody_mask_list.append(tmp_melody)
            length_list.append(curr_len)
            filename_list.append(seg['filename'])
            id_list.append(seg['id'])
            composer_list.append(seg['composer']) 
        
        batch_dict = {
            'id': torch.LongTensor(np.stack(id_list, axis=0)),
            'dec_inp': torch.LongTensor(np.stack(dec_inp_list, axis=0)),
            'dec_tgt': torch.LongTensor(np.stack(dec_tgt_list, axis=0)), 
            'inp_chord': torch.FloatTensor(np.stack(chord_mask_list, axis=0)),
            'inp_melody': torch.FloatTensor(np.stack(melody_mask_list, axis=0)),
            'length': torch.LongTensor(np.stack(length_list, axis=0)),
            'filename': filename_list,
            'composer': composer_list
        }
        return batch_dict