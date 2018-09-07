import logging
import torch
import torch.nn as nn
import numpy as np
import random
import pykp
from pykp.mask import GetMask, masked_softmax, TimeDistributedDense
from pykp.rnn_encoder import RNNEncoder
from pykp.rnn_decoder import RNNDecoder

class Seq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()

        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size
        #self.ctx_hidden_dim = opt.rnn_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout

        self.bridge = opt.bridge

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = opt.word2idx[pykp.io.PAD_WORD]
        self.pad_idx_trg = opt.word2idx[pykp.io.PAD_WORD]
        self.bos_idx = opt.word2idx[pykp.io.BOS_WORD]
        self.eos_token = opt.word2idx[pykp.io.EOS_WORD]
        self.unk_idx = opt.word2idx[pykp.io.UNK_WORD]

        self.share_embeddings = opt.share_embeddings

        '''
        self.attention_mode = opt.attention_mode    # 'dot', 'general', 'concat'
        self.input_feeding = opt.input_feeding

        self.copy_attention = opt.copy_attention    # bool, enable copy attention or not
        self.copy_mode = opt.copy_mode         # same to `attention_mode`
        self.copy_input_feeding = opt.copy_input_feeding
        self.reuse_copy_attn = opt.reuse_copy_attn
        self.copy_gate = opt.copy_gate

        self.must_teacher_forcing = opt.must_teacher_forcing
        self.teacher_forcing_ratio = opt.teacher_forcing_ratio
        self.scheduled_sampling = opt.scheduled_sampling
        self.scheduled_sampling_batches = opt.scheduled_sampling_batches
        self.scheduled_sampling_type = 'inverse_sigmoid'  # decay curve type: linear or inverse_sigmoid
        self.current_batch = 0  # for scheduled sampling

        self.device = opt.device

        if self.scheduled_sampling:
            logging.info("Applying scheduled sampling with %s decay for the first %d batches" % (self.scheduled_sampling_type, self.scheduled_sampling_batches))
        if self.must_teacher_forcing or self.teacher_forcing_ratio >= 1:
            logging.info("Training with All Teacher Forcing")
        elif self.teacher_forcing_ratio <= 0:
            logging.info("Training with All Sampling")
        else:
            logging.info("Training with Teacher Forcing with static rate=%f" % self.teacher_forcing_ratio)

        self.get_mask = GetMask(self.pad_idx_src)
        '''
        '''
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            self.pad_idx_src
        )
        '''
        self.encoder = RNNEncoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )

        self.decoder = RNNDecoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            coverage_attn=self.coverage_attn,
            copy_attn=self.copy_attn,
            pad_idx=self.pad_idx_trg,
            dropout=self.dropout
        )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = nn.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        if self.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        if not self.share_embeddings:
            self.decoder.embedding.weight.data.uniform_(-initrange, initrange)

        # TODO: model parameter init
        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        #self.encoder2decoder_hidden.bias.data.fill_(0)
        #self.encoder2decoder_cell.bias.data.fill_(0)
        #self.decoder2vocab.bias.data.fill_(0)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        batch_size, max_src_len = list(src.size())

        # Encoding
        '''
        src_emb = self.embedding(src)
        memory_bank, encoder_final_state = self.encoder(src_emb, src_lens)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])
        '''
        memory_bank, encoder_final_state = self.encoder(src, src_lens)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # Decoding
        h_t = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)
        #context = self.init_context(memory_bank)  # [batch, memory_bank_size]

        decoder_dist_all = []
        attention_dist_all = []

        if self.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float).requires_grad_()  # [batch, max_src_seq]
            #coverage_all = coverage.new_zeros((max_target_length, batch_size, max_src_len), dtype=torch.float)  # [max_trg_len, batch_size, max_src_len]
            coverage_all = []
        else:
            coverage = None
            coverage_all = None

        # init y_t to be BOS token
        y_t = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]

        #print(y_t[:5])

        for t in range(max_target_length):
            decoder_dist, h_t, _, attn_dist, p_gen, coverage = \
                self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            y_t = trg[:, t]
            #y_t_emb = trg_emb[:, t, :].unsqueeze(0)  # [1, batch, embed_size]

            #print(t)
        #print(trg_emb.size(1))

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
        if self.coverage_attn:
            coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
            assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        return decoder_dist_all, h_t, attention_dist_all, coverage_all

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state

    def init_context(self, memory_bank):
        # Init by max pooling, may support other initialization later
        context, _ = memory_bank.max(dim=1)
        return context
