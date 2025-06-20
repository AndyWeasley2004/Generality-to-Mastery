from torch import nn
import torch.nn.functional as F

from .vanilla_decoder import VanillaTransformerDecoder
from .transformer_helpers import (
  StructuredWordEmbedding,
  weights_init
)

class VanillaTransformer(nn.Module):
  def __init__(self, d_word_embed, event2word, dec_n_layer, dec_n_head, 
               dec_d_model, dec_d_ff, tgt_len, attn_type, 
               dec_dropout=0.1, pre_lnorm=False, adapter_positions=None):
    super(VanillaTransformer, self).__init__()

    self.d_word_embed = d_word_embed
    self.vocab_size = len(event2word) if 'PAD_None' in event2word.keys() else len(event2word) + 1

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout

    self.word_emb = StructuredWordEmbedding(event2word, d_word_embed, dec_d_model)    
    self.emb_dropout = nn.Dropout(dec_dropout)
    self.pad_index = self.word_emb.pad_idx

    self.decoder = VanillaTransformerDecoder(
                    dec_n_layer, dec_n_head, dec_d_model,
                    dec_d_ff, attn_type=attn_type, tgt_len=tgt_len, 
                    dropout=dec_dropout, pre_lnorm=pre_lnorm,
                    adapter_positions=adapter_positions)
    self.dec_out_proj = nn.Linear(dec_d_model, self.vocab_size)

    self.apply(weights_init)

  def generate(self, dec_input):
    dec_word_emb = self.word_emb(dec_input)
    dec_input = self.emb_dropout(dec_word_emb)
    dec_out = self.decoder(dec_input)
    dec_logits = self.dec_out_proj(dec_out)[-1, 0, :]
    return dec_logits

  def forward(self, dec_input):
    dec_word_emb = self.word_emb(dec_input)
    dec_input = self.emb_dropout(dec_word_emb)
    # print('[decoder input shape]', dec_input.shape)
    dec_out = self.decoder(dec_input)
    # print('[decoder output shape]', dec_out.shape)
    dec_logits = self.dec_out_proj(dec_out)
    # print('[logit shape]', dec_logits.shape)
    return dec_logits


  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    ce_loss = F.cross_entropy(
                    dec_logits.view(-1, dec_logits.size(-1)),
                    dec_tgt.contiguous().view(-1),
                    ignore_index=self.pad_index,
                    reduction=reduction
                  )

    return ce_loss