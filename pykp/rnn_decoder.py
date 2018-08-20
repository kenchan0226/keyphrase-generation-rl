import logging
import torch
import torch.nn as nn
from pykp.attention import Attention
import numpy as np
from pykp.masked_softmax import MaskedSoftmax
import math
import logging

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, coverage_attn, copy_attn, pad_idx, dropout=0.0):
        super(RNNDecoder, self).__init__()
        #self.input_size = input_size
        self.input_size = embed_size + memory_bank_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = dropout
        self.coverage_attn = coverage_attn
        self.copy_attn = copy_attn
        self.pad_token = pad_idx
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=False, batch_first=False, dropout=dropout)
        self.attention_layer = Attention(
            decoder_size=hidden_size,
            memory_bank_size=memory_bank_size,
            coverage_attn=coverage_attn
        )
        #self.p_gen_network = nn.Sequential(nn.Linear(input_size + hidden_size, 1), nn.Sigmoid())
        self.p_gen_linear = nn.Linear(self.input_size + hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        #self.p_gen_linear = nn.Linear(input_size + hidden_size, 1)
        #self.sigmoid = nn.Sigmoid()
        # self.vocab_dist_network = nn.Sequential(nn.Linear(hidden_size + memory_bank_size, hidden_size), nn.Linear(hidden_size, vocab_size), nn.Softmax(dim=1))
        self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size, hidden_size)
        self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, h, memory_bank, src_mask, context, max_num_oovs, src_oov, coverage):
        """
        :param y: [batch_size]
        :param h: [1, batch_size, decoder_size]
        :param memory_bank: [batch_size, max_src_seq_Len, memory_bank_size]
        :param src_mask: [batch_size, max_src_seq_len]
        :param context: [batch_size, memory_bank_size]
        :param max_num_oovs: int
        :param src_oov: [batch_size, max_src_seq_len]
        :param coverage: [batch_size, max_src_seq_len]
        :return:
        """
        batch_size, max_src_seq_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        assert h.size() == torch.Size([1, batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]
        # pass the concatenation of the input embedding and context vector to the RNN
        # insert one dimension to the context tensor
        rnn_input = torch.cat((y_emb, context.unsqueeze(0)), 2)  # [1, batch_size, embed_size + num_directions * encoder_size]
        _, h_next = self.rnn(rnn_input, h)  # [num_layers, batch, decoder_size]

        last_layer_h_next = h_next[-1,:,:]  # [batch, decoder_size]

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask, coverage)
        # context: [batch_size, memory_bank_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]

        # Debug
        #if math.isnan(attn_dist[0,0].item()):
        #    logging.info('nan attention distribution')

        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)  # [B, memory_bank_size + decoder_size]

        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.vocab_dist_linear_1(vocab_dist_input)))
        #logit_1 = self.vocab_dist_linear_1(vocab_dist_input)
        #logit_2 = self.vocab_dist_linear_2(logit_1)
        #vocab_dist = self.softmax(logit_2)

        p_gen = None
        if self.copy_attn:
            p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
            #p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))
            p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1-p_gen) * attn_dist

            if max_num_oovs > 0:
                #extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

            final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
        else:
            final_dist = vocab_dist

        assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        assert h_next.size() == torch.Size([1, batch_size, self.hidden_size])
        assert context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert attn_dist.size() == torch.Size([batch_size, max_src_seq_len])
        assert coverage.size() == torch.Size([batch_size, max_src_seq_len])

        return final_dist, h_next, context, attn_dist, p_gen, coverage


if __name__ == '__main__':
    embed_size = 30
    decoder_size = 100
    num_layers = 1
    memory_bank_size = 50
    input_size = embed_size + memory_bank_size
    vocab_size = 20
    coverage_attn = True
    copy_attn = True
    dropout = 0.0
    pad_idx = 0
    decoder = RNNDecoder(vocab_size, embed_size, decoder_size, num_layers, memory_bank_size, coverage_attn, copy_attn, pad_idx, dropout)
    batch_size = 5
    max_src_seq_len = 7

    #y_emb = torch.randn((1, batch_size, embed_size))
    y = torch.LongTensor(np.random.randint(1, 20, batch_size))
    h = torch.randn((num_layers, batch_size, decoder_size))
    memory_bank = torch.randn((batch_size, max_src_seq_len, memory_bank_size))
    context = torch.randn((batch_size, memory_bank_size))
    coverage = torch.rand((batch_size, max_src_seq_len))

    input_seq = np.random.randint(2, 20, (batch_size, max_src_seq_len))
    input_seq[batch_size - 1, max_src_seq_len - 1] = 0
    input_seq[batch_size - 1, max_src_seq_len - 2] = 0
    input_seq[batch_size - 2, max_src_seq_len - 1] = 0
    input_seq[1][5] = 1
    input_seq[3][2] = 1
    input_seq[3][5] = 1
    input_seq[3][6] = 1
    input_seq[0][2] = 1

    input_seq_oov = np.copy(input_seq)
    input_seq_oov[1][5] = 20
    input_seq_oov[3][2] = 20
    input_seq_oov[3][5] = 21
    input_seq_oov[3][6] = 22
    input_seq_oov[0][2] = 20

    input_seq = torch.LongTensor(input_seq)
    input_seq_oov = torch.LongTensor(input_seq_oov)

    src_mask = torch.ne(input_seq, 0)
    src_mask = src_mask.type(torch.FloatTensor)
    max_num_oovs = 3
    final_dist, h_next, context, attn_dist, p_gen, coverage = decoder(y, h, memory_bank, src_mask, context, max_num_oovs, input_seq_oov, coverage)
    print("Pass")

