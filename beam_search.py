"""
Adapted from
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
and seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""

import copy
import heapq
from queue import PriorityQueue

import sys
import torch

import pykp
from pykp.mask import GetMask
import numpy as np
import collections
import itertools
import logging
import penalties

from torch.distributions import Categorical

EPS = 1e-8

class Beam:
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):
        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                            .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        #self.stepwise_penalty = stepwise_penalty
        #self.block_ngram_repeat = block_ngram_repeat
        #self.exclusion_tokens = exclusion_tokens

    def get_current_tokens(self):
        """Get the outputs for the current timestep."""
        return self.next_ys[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def get_hyp(self, timestep, k):
        """
        walk back to construct the full hypothesis given the finished time step and beam idx
        :param timestep: int
        :param k: int
        :return:
        """
        hyp, attn = [], []
        # iterate from output sequence length (with eos but not bos) - 1 to 0
        for j in range(len(self.prev_ks[:timestep]) -1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])  # j+1 so that it will iterate from the <eos> token, and end before the <bos>
            attn.append(self.attn[j][k])  # since it does not has attn for bos, it will also iterate from the attn for <eos>
            # attn[j][k] Tensor with size [src_len]
            k = self.prev_ks[j][k]  # find the beam idx of the previous token

        # hyp[::-1]: a list of idx (zero dim tensor), with len = output sequence length
        # torch.stack(attn): FloatTensor, with size: [output sequence length, src_len]
        return hyp[::-1], torch.stack(attn)

    def advance(self, word_logits, attn_dist):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_logit`- probs of advancing from the last step [beam_size, vocab_size]
        * `attn_dist`- attention at the last step [beam_size, src_len]

        Returns: True if beam search is complete.
        """
        vocab_size = word_logits.size(1)
        # To be implemented: stepwise penalty

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_logits)):
                word_logits[k][self._eos] = -1e20
        # Sum the previous scores
        if len(self.prev_ks) > 0:
            beam_scores = word_logits + self.scores.unsqueeze(1).expand_as(word_logits)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
            # To be implemented: block n-gram repeated

        else:  # This is the first decoding step, every beam are the same
            beam_scores = word_logits[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_idx = flat_beam_scores.topk(self.size, 0, True, True)  # [beam_size]

        self.all_scores.append(self.score)  # list of tensor with size [beam_size]
        self.scores = best_scores

        # best_scores_idx indicate the idx in the flattened beam * vocab_isze array, so need to convert
        # the idx back to which beam and word each score came from.
        prev_k = best_scores_idx / vocab_size  # convert it to the beam indices that the top k scores came from, LongTensor, size: [beam_size]
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_idx - prev_k * vocab_size))  # convert it to the vocab indices, LongTensor, size: [beam_size]
        self.attn.append(attn_dist.index_select(0, prev_k))  # select the attention dist from the corresponding beam, size: [beam_size, src_len]
        self.global_scorer.update_global_state(self)  # update coverage vector, previous coverage penalty, and cov_total

        for i in range(self.next_ys[-1].size(0)): # For each generated token in the current step, check if it is EOS
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)  # compute the score penalize by length and coverage
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))  # penalized score, length of sequence, beam_idx

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs in the finished list
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys)-1, i)) # score, length of sequence (including <bos>, <eos>) -1, beam_idx
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t,k) for _, t, k in self.finished]
        return scores, ks

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
                 coverage_penalty='none'
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
        decoder_init_input = src.new_ones((batch_size * beam_size, 1)) * word2idx[pykp.io.BOS_WORD]  # [batch_size*beam_size, 1]

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

        beam_list = [Beam(beam_size, n_best=beam_size,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=self.pad_idx,
                                    eos=self.eos_idx,
                                    bos=self.bos_idx)
                for __ in range(batch_size)]

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
                    decoder_input.gt(self.model.vocab_size - 1), 0)

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
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, src_lens[batch_idx]])
                decoder_state = self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def _from_beam(self, beam_list):
        ret = {"predictions": [], "scores": [], "attention": []}
        for b in beam_list:
            n_best = self.beam_size
            scores, ks = b.sort_finished(minium=n_best)
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
        return decoder_state_transformed

class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(cov_penalty, length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores all the prediction scores of a beam based on penalty functions
        Return: normalized_probs, size: [beam_size]
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        """
        Keeps the coverage vector as sum of attentions
        """
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)  # [beam_size]
            beam.global_state["coverage"] = beam.attn[-1]  # [beam_size, src_len]
            self.cov_total = beam.attn[-1].sum(1)  # [beam_size], accumulate the penalty term for coverage
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])  # accumulate coverage vector

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty

