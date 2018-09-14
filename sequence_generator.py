"""
Adapted from
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
and seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""

import sys
import torch
import pykp
import logging
from beam import Beam
from beam import GNMTGlobalScorer

EPS = 1e-8

class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,
                 eos_idx,
                 bos_idx,
                 pad_idx,
                 beam_size,
                 max_sequence_length,
                 copy_attn=False,
                 coverage_attn=False,
                 include_attn_dist=True,
                 length_penalty_factor=0.0,
                 coverage_penalty_factor=0.0,
                 length_penalty='avg',
                 coverage_penalty='none',
                 cuda=True,
                 n_best=None
                 ):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
          eos_idx: the idx of the <eos> token
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          coverage_attn: use coverage attention or not
          include_attn_dist: include the attention distribution in the sequence obj or not.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
            alpha in: https://arxiv.org/abs/1609.08144
          length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
        """
        self.model = model
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_penalty_factor = length_penalty_factor
        self.coverage_penalty_factor = coverage_penalty_factor
        self.coverage_attn = coverage_attn
        self.include_attn_dist = include_attn_dist
        #self.lambda_coverage = lambda_coverage
        self.coverage_penalty = coverage_penalty
        self.copy_attn = copy_attn
        self.global_scorer = GNMTGlobalScorer(length_penalty_factor, coverage_penalty_factor, coverage_penalty, length_penalty)
        self.cuda = cuda
        if n_best is None:
            self.n_best = self.beam_size
        else:
            self.n_best = n_best

    def beam_search(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        """
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens)
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Init decoder state
        decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        # init initial_input to be BOS token
        decoder_init_input = src.new_ones((batch_size * beam_size, 1)) * self.bos_idx  # [batch_size*beam_size, 1]

        if self.coverage_attn:  # init coverage
            #coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, src_len]
            coverage = src.new_zeros((batch_size * beam_size, 1))  # [batch_size * beam_size ,1]
        else:
            coverage = None

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]

        beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx) for _ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                      .t().contiguous().view(-1))
            # decoder_input: [batch_size * beam_size]

            # Turn any copied words to UNKS
            if self.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _, coverage = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov, coverage)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def _from_beam(self, beam_list):
        ret = {"predictions": [], "scores": [], "attention": []}
        for b in beam_list:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            # Collect all the decoded sentences in to hyps (list of list of idx) and attn (list of tensor)
            for i, (times, k) in enumerate(ks[:n_best]):
                # Get the corresponding decoded sentence, and also the attn dist [seq_len, memory_bank_size].
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)  # 3d list of idx (zero dim tensor), with len [batch_size, n_best, output_seq_len]
            ret['scores'].append(scores)  # a 2d list of zero dim tensor, with len [batch_size, n_best]
            ret["attention"].append(attn)  # a 2d list of FloatTensor[output sequence length, src_len] , with len [batch_size, n_best]
            # hyp[::-1]: a list of idx (zero dim tensor), with len = output sequence length
            # torch.stack(attn): FloatTensor, with size: [output sequence length, src_len]
        return ret

    def beam_decoder_state_update(self, batch_idx, beam_indices, decoder_state):
        """
        :param batch_idx: int
        :param beam_indices: a long tensor of previous beam indices, size: [beam_size]
        :param decoder_state: [dec_layers, flattened_batch_size, decoder_size]
        :return:
        """
        decoder_layers, flattened_batch_size, decoder_size = list(decoder_state.size())
        assert flattened_batch_size % self.beam_size == 0
        original_batch_size = flattened_batch_size//self.beam_size
        # select the hidden states of a particular batch -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed = decoder_state.view(decoder_layers, self.beam_size, original_batch_size, decoder_size)[:, :, batch_idx]
        # select the hidden states of the beams specified by the beam_indices -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed.data.copy_(decoder_state_transformed.data.index_select(1, beam_indices))

    def sample(self, src, src_lens, src_oov, src_mask, oov_lists, max_sample_length, greedy=False):
        # src, src_lens, src_oov, src_mask, oov_lists, word2idx
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param max_sample_length: The max length of sequence that can be sampled by the model
        :param greedy: whether to sample the word with max prob at each decoding step
        :return:
        """
        batch_size, max_src_len = list(src.size())
        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # Init decoder state
        decoder_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        if self.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, max_src_seq]
        else:
            coverage = None

        # init y_t to be BOS token
        decoder_input = src.new_ones(batch_size) * self.bos_idx  # [batch_size]
        sample_list = [{"prediction": [], "attention": [], "done": False} for _ in range(batch_size)]
        log_selected_token_dist = []
        prediction_all = src.new_ones(batch_size, max_sample_length) * self.pad_idx

        for t in range(max_sample_length):
            # Turn any copied words to UNKS
            if self.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # [batch, vocab_size], [dec_layers, batch, decoder_size], [batch, memory_bank_size], [batch, src_len], [batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _, coverage = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov, coverage)

            if greedy:  # greedy decoding, only use in self-critical
                selected_token_dist, prediction = torch.max(decoder_dist, 1)
                log_selected_token_dist.append(torch.log(selected_token_dist + EPS))
            else:  # sampling according to the probability distribution from the decoder
                prediction = torch.multinomial(decoder_dist, 1)  # [batch, 1]
                # select the probability of sampled tokens, and then take log, size: [batch, 1], append to a list
                log_selected_token_dist.append(torch.log(decoder_dist + EPS).gather(1, prediction))

            for batch_idx, sample in enumerate(sample_list):
                if not sample['done']:
                    sample['prediction'].append(prediction[batch_idx][0])  # 0 dim tensor
                    sample['attention'].append(attn_dist[batch_idx])  # [src_len] tensor
                    if int(prediction[batch_idx][0].item()) == self.model.eos_idx:
                        sample['done'] = True
                else:
                    prediction[batch_idx][0].fill_(self.pad_idx)

            prediction_all[:, t] = prediction[:, 0]
            decoder_input = prediction[:, 0]  # [batch]

            if all((s['done'] for s in sample_list)):
                break

        log_selected_token_dist = torch.cat(log_selected_token_dist, dim=1)  # [batch, t]
        assert log_selected_token_dist.size() == torch.Size([batch_size, t])
        output_mask = torch.ne(prediction_all, self.pad_idx)[:, :t+1]  # [batch, t]
        output_mask = output_mask.type(torch.FloatTensor)
        assert output_mask.size() == log_selected_token_dist.size()

        return sample_list, log_selected_token_dist, output_mask
