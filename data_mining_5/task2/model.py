import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CodeNN(nn.Module):
    def __init__(self, config):
        super(CodeNN, self).__init__()
        self.conf = config
        self.margin = config['margin']

    def code_encoding(self, name, name_len, tokens, tok_len):

    def desc_encoding(self, desc, desc_len):

    def similarity(self, code_vec, desc_vec):
        return F.cosine_similarity(code_vec, desc_vec)

    def forward(self, name, name_len, tokens, tok_len, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        code_repr = self.code_encoding(name, name_len, tokens, tok_len)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr)  # [batch_sz x 1]

        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss
