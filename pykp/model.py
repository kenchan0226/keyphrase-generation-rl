import logging
import torch
import torch.nn as nn
import numpy as np
import random
import pykp
from pykp.mask import GetMask, masked_softmax, TimeDistributedDense
from pykp.rnn_encoder import *
from pykp.rnn_decoder import RNNDecoder
from pykp.target_encoder import TargetEncoder
from pykp.attention import Attention
from pykp.manager import ManagerBasic

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
        self.one2many_mode = opt.one2many_mode
        self.one2many = opt.one2many

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = opt.word2idx[pykp.io.PAD_WORD]
        self.pad_idx_trg = opt.word2idx[pykp.io.PAD_WORD]
        self.bos_idx = opt.word2idx[pykp.io.BOS_WORD]
        self.eos_idx = opt.word2idx[pykp.io.EOS_WORD]
        self.unk_idx = opt.word2idx[pykp.io.UNK_WORD]
        self.sep_idx = opt.word2idx[pykp.io.SEP_WORD]
        self.orthogonal_loss = opt.orthogonal_loss

        self.share_embeddings = opt.share_embeddings
        self.review_attn = opt.review_attn

        self.attn_mode = opt.attn_mode

        self.use_target_encoder = opt.use_target_encoder
        self.target_encoder_size = opt.target_encoder_size

        self.device = opt.device

        self.separate_present_absent = opt.separate_present_absent
        self.goal_vector_mode = opt.goal_vector_mode
        self.goal_vector_size = opt.goal_vector_size
        self.manager_mode = opt.manager_mode
        self.title_guided = opt.title_guided

        if self.separate_present_absent:
            self.peos_idx = opt.word2idx[pykp.io.PEOS_WORD]

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
        if self.title_guided:
            self.encoder = RNNEncoderTG(
                vocab_size=self.vocab_size,
                embed_size=self.emb_dim,
                hidden_size=self.encoder_size,
                num_layers=self.enc_layers,
                bidirectional=self.bidirectional,
                pad_token=self.pad_idx_src,
                dropout=self.dropout
            )
        else:
            self.encoder = RNNEncoderBasic(
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
            review_attn=self.review_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            dropout=self.dropout,
            use_target_encoder=self.use_target_encoder,
            target_encoder_size=self.target_encoder_size,
            goal_vector_mode=self.goal_vector_mode,
            goal_vector_size=self.goal_vector_size
        )

        if self.use_target_encoder:
            self.target_encoder = TargetEncoder(
                embed_size=self.emb_dim,
                hidden_size=self.target_encoder_size,
                vocab_size=self.vocab_size,
                pad_idx=self.pad_idx_trg
            )
            # use the same embedding layer as that in the decoder
            self.target_encoder.embedding.weight = self.decoder.embedding.weight
            self.target_encoder_attention = Attention(
                self.target_encoder_size,
                memory_bank_size=self.num_directions * self.encoder_size,
                coverage_attn=False,
                attn_mode="general"
            )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = nn.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        if self.separate_present_absent and self.goal_vector_mode > 0:
            if self.manager_mode == 2:  # use GRU as a manager
                self.manager = nn.GRU(input_size=self.decoder_size, hidden_size=self.goal_vector_size, num_layers=1, bidirectional=False, batch_first=False, dropout=self.dropout)
                self.bridge_manager = opt.bridge_manager
                if self.bridge_manager:
                    self.manager_bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.goal_vector_size)
                else:
                    self.manager_bridge_layer = None
            elif self.manager_mode == 1:  # use two trainable vectors only
                self.manager = ManagerBasic(self.goal_vector_size)

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

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, num_trgs=None, sampled_source_representation_2dlist=None, source_representation_target_list=None, title=None, title_lens=None, title_mask=None):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param num_trgs: only effective in one2many mode 2, a list of num of targets in each batch, with len=batch_size
        :param sampled_source_representation_2dlist: only effective when using target encoder, a 2dlist of tensor with dim=[memory_bank_size]
        :param source_representation_target_list: a list that store the index of ground truth source representation for each batch, dim=[batch_size]
        :return:
        """
        batch_size, max_src_len = list(src.size())

        # Encoding
        memory_bank, encoder_final_state = self.encoder(src, src_lens, src_mask, title, title_lens, title_mask)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        if self.one2many and self.one2many_mode > 1:
            assert num_trgs is not None, "If one2many mode is 2, you must supply the number of targets in each sample."
            assert len(num_trgs) == batch_size, "The length of num_trgs is incorrect"

        if self.use_target_encoder and sampled_source_representation_2dlist is not None:
            # put the ground-truth encoder representation, need to call detach() first
            for i in range(batch_size):
                sampled_source_representation_2dlist[i][source_representation_target_list[i]] = encoder_final_state[i, :].detach()  # [memory_bank_size]
            source_representation_sample_size = len(sampled_source_representation_2dlist[0])
            sampled_source_representation = self.tensor_2dlist_to_tensor(
                sampled_source_representation_2dlist, batch_size, self.num_directions * self.encoder_size, [source_representation_sample_size]*batch_size)
            sampled_source_representation = torch.transpose(sampled_source_representation, 1, 2).contiguous()
            assert sampled_source_representation.size() == torch.Size([batch_size, source_representation_sample_size, self.num_directions * self.encoder_size])
            # sampled_source_representation: [batch_size, source_representation_sample_size, memory_bank_size]

        # Decoding
        h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
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

        if self.review_attn:
            decoder_memory_bank = h_t_init[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            assert decoder_memory_bank.size() == torch.Size([batch_size, 1, self.decoder_size])
        else:
            decoder_memory_bank = None

        if self.orthogonal_loss:  # create a list of batch_size empty list
            delimiter_decoder_states_2dlist = [[] for i in range(batch_size)]

        if self.use_target_encoder:
            # init the hidden state of target encoder to zero vector
            h_te_t = h_t_init.new_zeros(1, batch_size, self.target_encoder_size)
            # create a list of batch_size empty list
            delimiter_target_encoder_states_2dlist = [[] for i in range(batch_size)]

        # init y_t to be BOS token
        #y_t = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]
        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]

        if self.separate_present_absent and self.goal_vector_mode > 0:
            # byte tensor with size=batch_size to keep track of which batch has been proceeded to absent prediction
            is_absent = torch.zeros(batch_size, dtype=torch.uint8)

        #print(y_t[:5])
        '''
        for t in range(max_target_length):
            # determine the hidden state that will be feed into the next step
            # according to the time step or the target input
            re_init_indicators = (y_t == self.sep_idx)  # [batch]

            if t == 0:
                h_t = h_t_init
            elif self.one2many_mode == 2 and re_init_indicators.sum().item() != 0:
                h_t = []
                # h_t_next [dec_layers, batch_size, decoder_size]
                # h_t_init [dec_layers, batch_size, decoder_size]
                for batch_idx, indicator in enumerate(re_init_indicators):
                    if indicator.item() == 0:
                        h_t.append(h_t_next[:, batch_idx, :].unsqueeze(1))
                    else:
                        # some examples complete one keyphrase
                        h_t.append(h_t_init[:, batch_idx, :].unsqueeze(1))
                h_t = torch.cat(h_t, dim=1)  # [dec_layers, batch_size, decoder_size]
            else:
                h_t = h_t_next

            decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            y_t = trg[:, t]
            #y_t_emb = trg_emb[:, t, :].unsqueeze(0)  # [1, batch, embed_size]
        '''
            #print(t)
        #print(trg_emb.size(1))

        #pred_counters = trg.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]

        for t in range(max_target_length):
            # determine the hidden state that will be feed into the next step
            # according to the time step or the target input
            #re_init_indicators = (y_t == self.eos_idx)  # [batch]
            if t == 0:
                pred_counters = trg.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]
            else:
                re_init_indicators = (y_t_next == self.eos_idx)  # [batch_size]
                pred_counters += re_init_indicators

            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
                #re_init_indicators = (y_t == self.eos_idx)  # [batch]
                #pred_counters = re_init_indicators
                #pred_counters = trg.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]

            elif self.one2many and self.one2many_mode == 2 and re_init_indicators.sum().item() > 0:
                #re_init_indicators = (y_t_next == self.eos_idx)  # [batch]
                #pred_counters += re_init_indicators
                h_t = []
                y_t = []
                # h_t_next [dec_layers, batch_size, decoder_size]
                # h_t_init [dec_layers, batch_size, decoder_size]
                for batch_idx, (indicator, pred_count, trg_count) in enumerate(zip(re_init_indicators, pred_counters, num_trgs)):
                    if indicator.item() == 1 and pred_count.item() < trg_count:
                        # some examples complete one keyphrase
                        h_t.append(h_t_init[:, batch_idx, :].unsqueeze(1))
                        y_t.append(y_t_init[batch_idx].unsqueeze(0))
                    else:  # indicator.item() == 0 or indicator.item() == 1 and pred_count.item() == trg_count:
                        h_t.append(h_t_next[:, batch_idx, :].unsqueeze(1))
                        y_t.append(y_t_next[batch_idx].unsqueeze(0))
                h_t = torch.cat(h_t, dim=1)  # [dec_layers, batch_size, decoder_size]
                y_t = torch.cat(y_t, dim=0)  # [batch_size]
            elif self.one2many and self.one2many_mode == 3 and re_init_indicators.sum().item() > 0:
                # re_init_indicators = (y_t_next == self.eos_idx)  # [batch]
                # pred_counters += re_init_indicators
                h_t = h_t_next
                y_t = []
                # h_t_next [dec_layers, batch_size, decoder_size]
                # h_t_init [dec_layers, batch_size, decoder_size]
                for batch_idx, (indicator, pred_count, trg_count) in enumerate(
                        zip(re_init_indicators, pred_counters, num_trgs)):
                    if indicator.item() == 1 and pred_count.item() < trg_count:
                        # some examples complete one keyphrase
                        y_t.append(y_t_init[batch_idx].unsqueeze(0))
                    else:  # indicator.item() == 0 or indicator.item() == 1 and pred_count.item() == trg_count:
                        y_t.append(y_t_next[batch_idx].unsqueeze(0))
                y_t = torch.cat(y_t, dim=0)  # [batch_size]
            else:
                h_t = h_t_next
                y_t = y_t_next

            if self.review_attn:
                if t > 0:
                    decoder_memory_bank = torch.cat([decoder_memory_bank, h_t[-1, :, :].unsqueeze(1)], dim=1)  # [batch, t+1, decoder_size]

            if self.use_target_encoder:
                # encode the previous token
                h_te_t_next = self.target_encoder(y_t.detach(), h_te_t)
                h_te_t = h_te_t_next  # [1, batch_size, target_encoder_size]
                # decoder_input = (y_t, h_te_t)
                # if this target encoder state corresponds to the delimiter, stack it
                for i in range(batch_size):
                    if y_t[i].item() == self.sep_idx:
                        delimiter_target_encoder_states_2dlist[i].append(h_te_t[0, i, :])  # [target_encoder_size]
            else:
                h_te_t = None
                # decoder_input = y_t

            if self.separate_present_absent and self.goal_vector_mode > 0:
                # update the is_absent vector
                for i in range(batch_size):
                    if y_t[i].item() == self.peos_idx:
                        is_absent[i] = 1
                #
                if self.manager_mode == 1:
                    g_t = self.manager(is_absent)
            else:
                g_t = None

            decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage, decoder_memory_bank, h_te_t, g_t)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            y_t_next = trg[:, t]  # [batch]

            # if this hidden state corresponds to the delimiter, stack it
            if self.orthogonal_loss:
                for i in range(batch_size):
                    if y_t_next[i].item() == self.sep_idx:
                        delimiter_decoder_states_2dlist[i].append(h_t_next[-1, i, :])  # [decoder_size]

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

        # Pad delimiter_decoder_states_2dlist with zero vectors
        if self.orthogonal_loss:
            assert len(delimiter_decoder_states_2dlist) == batch_size
            delimiter_decoder_states_lens = [len(delimiter_decoder_states_2dlist[i]) for i in range(batch_size)]
            # [batch_size, decoder_size, max_num_delimiters]
            delimiter_decoder_states = self.tensor_2dlist_to_tensor(delimiter_decoder_states_2dlist, batch_size, self.decoder_size, delimiter_decoder_states_lens)
            """
            max_num_delimiters = max(delimiter_decoder_states_lens)
            for i in range(batch_size):
                for j in range(max_num_delimiters - delimiter_decoder_states_lens[i]):
                    delimiter_decoder_states_2dlist[i].append(torch.zeros_like(h_t_next[-1, 0, :]))  # [decoder_size]
                delimiter_decoder_states_2dlist[i] = torch.stack(delimiter_decoder_states_2dlist[i], dim=1)  # [decoder_size, max_num_delimiters]
            delimiter_decoder_states = torch.stack(delimiter_decoder_states_2dlist, dim=0)  # [batch_size, deocder_size, max_num_delimiters]
            """
        else:
            delimiter_decoder_states_lens = None
            delimiter_decoder_states = None

        # Pad the target_encoder_states_2dlist with zero vectors
        if self.use_target_encoder and sampled_source_representation_2dlist is not None:
            assert len(delimiter_target_encoder_states_2dlist) == batch_size
            # Pad the delimiter_target_encoder_states_2dlist with zeros and convert it to a tensor
            delimiter_target_encoder_states_lens = [len(delimiter_target_encoder_states_2dlist[i]) for i in range(batch_size)]
            # [batch_size, target_encoder_size, max_num_delimiters]
            delimiter_target_encoder_states = self.tensor_2dlist_to_tensor(delimiter_target_encoder_states_2dlist, batch_size, self.target_encoder_size, delimiter_target_encoder_states_lens)
            max_num_delimiters = delimiter_target_encoder_states.size(2)
            # Perform attention step by step
            source_classification_dist_all = []
            for i in range(max_num_delimiters):
                # delimiter_target_encoder_states[:, :, i]: [batch_size, target_encoder_size]
                # sampled_source_representation: [batch_size, source_representation_sample_size, memory_bank_size]
                _, source_classification_dist, _ = self.target_encoder_attention(delimiter_target_encoder_states[:, :, i], sampled_source_representation)
                # source_classification_dist: [batch_size, source_representation_sample_size]
                source_classification_dist_all.append(source_classification_dist.unsqueeze(1))  # [batch_size, 1, source_representation_sample_size]
            source_classification_dist_all = torch.cat(source_classification_dist_all, dim=1)  # [batch_size, max_num_delimiters, source_representation_sample_size]
        else:
            source_classification_dist_all = None

        return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all, delimiter_decoder_states, delimiter_decoder_states_lens, source_classification_dist_all

    def tensor_2dlist_to_tensor(self, tensor_2d_list, batch_size, hidden_size, seq_lens):
        """
        :param tensor_2d_list: a 2d list of tensor with size=[hidden_size], len(tensor_2d_list)=batch_size, len(tensor_2d_list[i])=seq_len[i]
        :param batch_size:
        :param hidden_size:
        :param seq_lens: a list that store the seq len of each batch, with len=batch_size
        :return: [batch_size, hidden_size, max_seq_len]
        """
        # assert tensor_2d_list[0][0].size() == torch.Size([hidden_size])
        max_seq_len = max(seq_lens)
        for i in range(batch_size):
            for j in range(max_seq_len - seq_lens[i]):
                tensor_2d_list[i].append( torch.ones(hidden_size).to(self.device) * self.pad_idx_trg )  # [hidden_size]
            tensor_2d_list[i] = torch.stack(tensor_2d_list[i], dim=1)  # [hidden_size, max_seq_len]
        tensor_3d = torch.stack(tensor_2d_list, dim=0)  # [batch_size, hidden_size, max_seq_len]
        return tensor_3d

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
