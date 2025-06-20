import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import re

def generate_causal_mask(seq_len, device):
    mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask.bool()

def generate_bidirectional_pad_mask(max_seqlen, batch_lens):
    mask = torch.zeros(len(batch_lens), max_seqlen, dtype=bool)
    for i, l in enumerate(batch_lens):
        mask[i, l:] = True
    return mask

def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)

def weight_init_orthogonal(weight, gain):
  nn.init.orthogonal_(weight, gain)

def bias_init(bias):
  nn.init.constant_(bias, 0.0)
  
def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)


class WordEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, emb_scale=0.5, pad_idx=None):
        super(WordEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj ** emb_scale

        if pad_idx is None:
            pad_idx = n_token - 1
            
        self.emb_lookup = nn.Embedding(n_token, d_embed, padding_idx=pad_idx)
        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None

    def forward(self, inp_tokens):
        inp_emb = self.emb_lookup(inp_tokens)
        
        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)

        return inp_emb.mul_(self.emb_scale)


class StructuredWordEmbedding(nn.Module):
    def __init__(self, event2word, d_embed, d_proj, emb_scale=0.5, pad_idx=None):
        super().__init__()

        self.event2word = event2word
        self.n_token = len(event2word) if 'PAD_None' in event2word else len(event2word) + 1
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj ** emb_scale

        if pad_idx is None:
            if 'PAD_None' in event2word:
                pad_idx = event2word['PAD_None']
            else:
                pad_idx = self.n_token - 1
        self.pad_idx = pad_idx

        self.emb_lookup = nn.Embedding(self.n_token, d_embed, padding_idx=pad_idx)

        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None
        self.initialize_embeddings(event2word)

    @staticmethod
    def get_note_embedding(family_base, pitch, scale=0.01, pitch_scale=0.001, pitch_direction=None):
        if pitch_direction is None:
            pitch_direction = torch.randn_like(family_base)

        perturbation = torch.randn_like(family_base) * scale
        pitch_offset = (pitch - 60.0) * pitch_scale * pitch_direction
        return family_base + perturbation + pitch_offset

    @staticmethod
    def get_time_shift_embedding(family_base, shift_val, scale=0.01, shift_scale=0.001, shift_direction=None):
        if shift_direction is None:
            shift_direction = torch.randn_like(family_base)

        perturbation = torch.randn_like(family_base) * scale
        magnitude = torch.log(torch.tensor([shift_val + 1], dtype=torch.float))
        shift_offset = magnitude * shift_scale * shift_direction
        return family_base + perturbation + shift_offset

    def initialize_embeddings(self, event2word):
        """
        Assign a structured initial embedding to each token in the vocabulary.
        """
        with torch.no_grad():
            embedding_weights = self.emb_lookup.weight
            note_on_tokens =  {k: v for k, v in event2word.items() if k.startswith('Note_On')}
            note_off_tokens = {k: v for k, v in event2word.items() if k.startswith('Note_Off')}
            shift_tokens =    {k: v for k, v in event2word.items() if k.startswith('Duration')}

            family_bases = {}
            for family_name in ["Note_On", "Note_Off", "Duration"]:
                base_vec = torch.randn(self.d_embed) * 0.02
                family_bases[family_name] = base_vec

            pitch_direction = torch.randn(self.d_embed)
            pitch_direction /= pitch_direction.norm() + 1e-9

            shift_direction = torch.randn(self.d_embed)
            shift_direction /= shift_direction.norm() + 1e-9

            # Initialize Note_On tokens
            for token_str, idx in note_on_tokens.items():
                parts = token_str.split("_")
                pitch = int(parts[2])
                token_vec = self.get_note_embedding(
                    family_bases["Note_On"],
                    pitch=pitch,
                    scale=0.01,
                    pitch_scale=0.001,
                    pitch_direction=pitch_direction
                )
                embedding_weights[idx] = token_vec

            # Initialize Note_Off tokens
            for token_str, idx in note_off_tokens.items():
                parts = token_str.split("_")
                pitch = int(parts[2])
                token_vec = self.get_note_embedding(
                    family_bases["Note_Off"],
                    pitch=pitch,
                    scale=0.01,
                    pitch_scale=0.001,
                    pitch_direction=pitch_direction
                )
                embedding_weights[idx] = token_vec

            # Initialize Time_Shift tokens
            for token_str, idx in shift_tokens.items():
                parts = token_str.split("_")
                shift_val_str = parts[-1]
                shift_val = int(shift_val_str)
                token_vec = self.get_time_shift_embedding(
                    family_bases["Duration"],
                    shift_val=shift_val,
                    scale=0.01,
                    shift_scale=0.001,
                    shift_direction=shift_direction
                )
                embedding_weights[idx] = token_vec

            norms = embedding_weights.norm(dim=-1, keepdim=True) + 1e-6
            embedding_weights.div_(norms)

    def forward(self, inp_tokens):
        inp_emb = self.emb_lookup(inp_tokens)

        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)

        return inp_emb.mul_(self.emb_scale)


def get_min_max_pitch_idx(idx2event):
    min_idx, max_idx = len(idx2event), 0

    for k, v in idx2event.items():
        if 'Note_Pitch' in v:
            min_idx = min(min_idx, k)
            max_idx = max(max_idx, k)
    
    return min_idx, max_idx


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]