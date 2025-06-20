import torch
from torch import nn
import torch.nn.functional as F
import math

from .transformer_helpers import PositionalEmbedding, weights_init, StructuredWordEmbedding

class VanillaTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1, pre_lnorm=False):
        super(VanillaTransformerEncoderLayer, self).__init__()
        self.pre_lnorm = pre_lnorm
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head,
            dropout=dropout, batch_first=False, bias=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        """
        x: (seq_len, batch_size, d_model)
        key_padding_mask: (batch_size, seq_len) where True indicates padded positions.
        """
        if self.pre_lnorm:
            x_norm = self.layernorm1(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            x = x + attn_out
            x_norm2 = self.layernorm2(x)
            ffn_out = self.ffn(x_norm2)
            x = x + ffn_out
        else:
            attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.layernorm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.layernorm2(x + ffn_out)
        return x

class VanillaTransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, max_seq_len, dropout=0.1, pre_lnorm=False):
        super(VanillaTransformerEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            VanillaTransformerEncoderLayer(d_model, d_ff, n_head, dropout=dropout, pre_lnorm=pre_lnorm)
            for _ in range(n_layer)
        ])
        self.pos_emb = PositionalEmbedding(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, enc_input, key_padding_mask=None):
        """
        enc_input: (seq_len, batch_size, d_model)
        key_padding_mask: (batch_size, seq_len)
        """
        seq_len, bsz, _ = enc_input.size()
        # Create positional embedding for all positions.
        pos_seq = torch.arange(seq_len, device=enc_input.device, dtype=torch.float)
        pos_emb = self.pos_emb(pos_seq, bsz=bsz)
        x = enc_input + pos_emb
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

##########################################
# Composer Classifier Model
##########################################
class ComposerClassifier(nn.Module):
    def __init__(self, d_word_embed, event2word, n_layer, n_head, 
                 d_model, d_ff, max_seq_len, num_classes, dropout=0.1, pre_lnorm=False):
        """
        d_word_embed: dimension of word embedding
        event2word: vocabulary dictionary
        num_classes: number of composer classes (e.g. len(used_categories))
        """
        super(ComposerClassifier, self).__init__()
        self.word_emb = StructuredWordEmbedding(event2word, d_word_embed, d_model)
        self.pad_index = self.word_emb.pad_idx
        self.emb_dropout = nn.Dropout(dropout)
        self.encoder = VanillaTransformerEncoder(n_layer, n_head, d_model, d_ff, max_seq_len, dropout, pre_lnorm)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.apply(weights_init)

    def forward(self, inp_tokens):
        """
        inp_tokens: (batch_size, seq_len)  [assumed to be LongTensor]
        """
        # Get word embeddings; shape: (batch_size, seq_len, d_model)
        emb = self.word_emb(inp_tokens)
        emb = self.emb_dropout(emb)
        # Transformer modules expect (seq_len, batch_size, d_model)
        emb = emb.transpose(0, 1)
        # Create key_padding_mask: True for positions equal to pad_index.
        key_padding_mask = (inp_tokens == self.pad_index)
        # Pass through the encoder.
        enc_out = self.encoder(emb, key_padding_mask=key_padding_mask)
        # Use the first token's representation as the “CLS” embedding.
        cls_token_rep = enc_out[0]  # (batch_size, d_model)
        logits = self.cls_head(cls_token_rep)  # (batch_size, num_classes)
        return logits
