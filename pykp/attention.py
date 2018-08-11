import logging
import torch
import torch.nn as nn
import numpy as np
from pykp.masked_softmax import MaskedSoftmax

class Attention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, coverage_attn):
        super(Attention, self).__init__()
        # attention
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.coverage_attn = coverage_attn
        if coverage_attn:
            self.coverage_project = nn.Linear(1, decoder_size, bias=False)
        self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.v = nn.Linear(decoder_size, 1, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, decoder_state, memory_bank, src_mask, coverage=None):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :param coverage: [batch_size, max_input_seq_len]
        :return: context: [batch_size, self.num_directions * self.encoder_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        """
        # init dimension info
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        # project memory_bank
        memory_bank = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
        encoder_feature = self.memory_project(memory_bank)  # [batch_size*max_input_seq_len, decoder size]

        # project decoder state
        dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
        dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len, decoder_size).contiguous()
        dec_feature_expanded = dec_feature_expanded.view(-1, decoder_size)  # [batch_size*max_input_seq_len, decoder_size]

        # sum up attention features
        att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

        # Apply coverage
        if self.coverage_attn:
            coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
            coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
            att_features = att_features + coverage_feature

        # compute attention score and normalize them
        e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
        scores = self.v(e)  # [batch_size*max_input_seq_len, 1]
        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        attn_dist = self.softmax(scores, mask=src_mask)

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1) # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_input_seq_len, memory_bank_size)  # batch_size, max_input_seq_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        # Update coverage
        if self.coverage_attn:
            coverage = coverage.view(-1, max_input_seq_len)
            coverage = coverage + attn_dist
            assert coverage.size() == torch.Size([batch_size, max_input_seq_len])

        assert attn_dist.size() == torch.Size([batch_size, max_input_seq_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        return context, attn_dist, coverage

if __name__=='__main__':
    #decoder_size, memory_bank_size, coverage_attn
    batch_size = 5
    decoder_size = 100
    memory_bank_size = 100
    coverage_attn = True
    attenton_layer = Attention(decoder_size, memory_bank_size, coverage_attn)
    max_input_seq_len = 7
    decoder_state = torch.randn((batch_size, decoder_size))
    memory_bank = torch.randn((batch_size, max_input_seq_len, memory_bank_size))

    input_seq = np.ones((batch_size, max_input_seq_len))
    input_seq[batch_size-1,max_input_seq_len-1] = 0
    input_seq[batch_size-1,max_input_seq_len-2] = 0
    input_seq[batch_size-2,max_input_seq_len-1] = 0
    input_seq = torch.LongTensor(input_seq)
    enc_padding_mask = torch.ne(input_seq, 0)
    enc_padding_mask = enc_padding_mask.type(torch.FloatTensor)
    coverage = torch.zeros((batch_size, max_input_seq_len))

    context, attn_dist, coverage = attenton_layer(decoder_state, memory_bank, enc_padding_mask, coverage)
