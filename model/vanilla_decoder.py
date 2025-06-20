import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer_helpers import PositionalEmbedding


class VanillaTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, tgt_len, 
                 dropout=0.1, pre_lnorm=False, attn_type='mha'):
        super().__init__()
        if attn_type not in ['mha', 'rel']:
            print('Not supported attention types')
            exit()

        self.pre_lnorm = pre_lnorm
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head,
                                            dropout=dropout, batch_first=False, 
                                            bias=False)  # shape: (bsz, seq, dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        """
        x: (bsz, seq_len, d_model)
        """
        if self.pre_lnorm:
            # pre-LN
            x_norm = self.layernorm1(x)
            attn_out,_ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            x = x + attn_out
            x_norm2 = self.layernorm2(x)
            ffn_out = self.ffn(x_norm2)
            x = x + ffn_out
        else:
            # post-LN
            attn_out,_ = self.self_attn(x, x, x, attn_mask=attn_mask)
            x = self.layernorm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.layernorm2(x + ffn_out)

        return x


class GatedStyleAdapter(nn.Module):
    def __init__(self, d_model):
        super(GatedStyleAdapter, self).__init__()
        self.linear1 = nn.Linear(d_model * 2, d_model * 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x, global_cond):
        # Expand the global condition to match x's sequence length
        global_cond_expanded = global_cond.expand(x.size(0), -1, -1)
        x_cat = torch.cat([x, global_cond_expanded], dim=-1)
        hidden = self.gelu(self.linear1(x_cat))
        output = self.linear2(hidden)
        return output + x  # residual connection


class VanillaTransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, attn_type,
                 tgt_len=2400, dropout=0.1, pre_lnorm=False, adapter_positions=None):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer

        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            VanillaTransformerDecoderLayer(
                d_model, d_ff, n_head, tgt_len,
                dropout=dropout, attn_type=attn_type,
                pre_lnorm=pre_lnorm
            )
            for _ in range(n_layer)
        ])

        self.pos_emb = PositionalEmbedding(d_model)

        # Create adapters at specified positions.
        self.adapter_positions = adapter_positions if adapter_positions is not None else []
        self.adapters = nn.ModuleDict()
        for pos in self.adapter_positions:
            if pos < 0 or pos >= n_layer:
                raise ValueError(f"Adapter position {pos} is invalid. It must be between 0 and {n_layer-1}.")
            self.adapters[str(pos)] = GatedStyleAdapter(d_model)

    def forward(self, dec_input):
        seq_len, bsz, _ = dec_input.size()

        # Create an upper-triangular attention mask.
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=dec_input.device, dtype=torch.bool),
            diagonal=1
        )

        # Extract the global style condition from the first token's embedding.
        global_cond = dec_input[0:1, :, :]
        x = self.drop(dec_input)

        # Add positional embeddings.
        pos_seq = torch.arange(seq_len, device=dec_input.device, dtype=torch.float)
        pos_emb = self.pos_emb(pos_seq, bsz=bsz)
        x = x + pos_emb

        # Process each decoder layer.
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
            
            if str(i) in self.adapters:
                # print('adapter runned at layer', i)
                # _= input()
                x = self.adapters[str(i)](x, global_cond)

        core_out = self.drop(x)
        return core_out