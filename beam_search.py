"""
Class for generating sequences
Adapted from
https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/sequence_generator.py
https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/beam_search.py
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


from torch.distributions import Categorical

EPS = 1e-8


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, batch_idx, word_idx_list, decoder_state, memory_bank, src_mask, src_oov, oov_list, log_probs, score, context, coverage=None, attn_dist=None):
        """Initializes the Sequence.

        Args:
          batch_id: Original id of batch
          word_idx_list: List of word ids in the sequence.
          dec_hidden: model hidden state after generating the previous word.
          context: attention read out
          log_probs:  The log-probability of each word in the sequence.
          score:    Score of the sequence (log-probability)
        """
        self.batch_idx = batch_idx
        self.word_idx_list = word_idx_list
        self.vocab = set(word_idx_list)  # for filtering duplicates
        self.decoder_state = decoder_state
        self.memory_bank = memory_bank
        self.src_mask = src_mask
        self.src_oov = src_oov
        self.oov_list = oov_list
        self.log_probs = log_probs
        self.score = score
        self.context = context
        self.coverage = coverage
        self.attn_dist = attn_dist

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN_heap(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def __len__(self):
        assert self._data is not None
        return len(self._data)

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,
                 eos_idx,
                 beam_size,
                 max_sequence_length,
                 coverage_attn=False,
                 include_attn_dist=True,
                 length_normalization_factor=0.0,
                 length_normalization_const=5.
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
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_normalization_factor = length_normalization_factor
        self.length_normalization_const = length_normalization_const
        self.coverage_attn = coverage_attn
        self.include_attn_dist = include_attn_dist


    def sequence_to_batch(self, sequence_lists):
        '''
        Convert 2d sequences (batch_size, beam_size) into 1d batches
        :param sequence_lists: a list of TopN_heap object, each TopN_heap object contains beam_size of sequences
        :return:
        '''
        # [list of batch idx of seq in topN_heap_1, list of batch idx of seq in topN_heap_2, ...]
        seq_idx2batch_idx = [[seq.batch_idx for seq in sequence_list.extract()] for sequence_list in sequence_lists]

        # to easily map the partial_sequences back to the flattened_sequences
        # flatten_id_map = [list of seq_id in topN_heap_1, list of seq_id in topN_heap_2, ...]
        # seq_id increment from 0 at the first seq in first batch up to the end
        seq_idx = 0
        flattened_idx_map = []
        for sequence_list in sequence_lists:
            seq_indices = []
            for seq in sequence_list.extract():
                seq_indices.append(seq_idx)
                seq_idx += 1
            flattened_idx_map.append(seq_indices)

        # flatten sequence_list into [ seq_obj_1, seq_obj_2, ...], with len = batch_size * beam_size = flattened_batch_size
        flattened_sequences = list(itertools.chain(*[seq.extract() for seq in sequence_lists]))
        flattened_batch_size = len(flattened_sequences)

        # concatenate each token generated at the previous time step into a tensor, [flattened_batch_size, 1]
        # if the token is a oov, replace it with <unk>
        inputs = torch.cat([torch.LongTensor([seq.word_idx_list[-1]] if seq.word_idx_list[-1] < self.model.vocab_size else [self.model.unk_idx]) for seq in flattened_sequences])
        inputs = inputs.to(flattened_sequences[0].memory_bank.device)
        #assert inputs.size() == torch.Size([flattened_batch_size, 1])
        assert inputs.size() == torch.Size([flattened_batch_size])
        # concat each hidden state in flattened_sequences into a tensor
        decoder_states = torch.cat([seq.decoder_state for seq in flattened_sequences], dim=1) # [dec_layers, flattened_batch_size, decoder_size]
        '''
        if isinstance(flattened_sequences[0].dec_hidden, tuple):  
            h_states = torch.cat([seq.dec_hidden[0] for seq in flattened_sequences]).view(1, batch_size, -1)
            c_states = torch.cat([seq.dec_hidden[1] for seq in flattened_sequences]).view(1, batch_size, -1)
            dec_hiddens = (h_states, c_states)
        else:
            dec_hiddens = torch.cat([seq.state for seq in flattened_sequences])
        '''

        # concat each memory_bank, src_mask, src_oovs, into tensors, concat oov_lists into a 2D list
        memory_banks = torch.cat([seq.memory_bank for seq in flattened_sequences], dim=0)  # [flatten_batch_size, src_len, memory_bank_size]
        src_masks = torch.cat([seq.src_mask for seq in flattened_sequences], dim=0)  # [flatten_batch_size, src_len]
        src_oovs = torch.cat([seq.src_oov for seq in flattened_sequences], dim=0)  # [flatten_batch_size, src_len]
        coverages = torch.cat([seq.coverage for seq in flattened_sequences], dim=0) if self.coverage_attn else None  # [flatten_batch_size, src_len]
        contexts = torch.cat([seq.context for seq in flattened_sequences], dim=0)  # [flatten_batch_size, memory_bank_size]
        oov_lists = [seq.oov_list for seq in flattened_sequences]

        return seq_idx2batch_idx, flattened_idx_map, inputs, decoder_states, memory_banks, src_masks, src_oovs, contexts, coverages, oov_lists

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

        #with torch.no_grad():
        batch_size = src.size(0)

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens)
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]

        # Init decoder state
        decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        # init initial_input to be BOS token
        decoder_init_input = src.new_ones(batch_size) * word2idx[pykp.io.BOS_WORD]  # [batch_size]

        # init context
        context = self.model.init_context(memory_bank)  # [batch, memory_bank_size]

        if self.coverage_attn:  # init coverage
            coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, src_len]

        # maintain a TopN heaps for each batch, each topN heaps has a capacity of beam_size, used to store sequence objects
        partial_sequences = [TopN_heap(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN_heap(sys.maxsize) for _ in range(batch_size)]

        # store a sequence object to each TopN_heap in partial sequences with initial input, initial states, etc.
        # partial_sequences, a list of topN heaps, len=batch_size
        for batch_i in range(batch_size):
            seq = Sequence(
                batch_idx=batch_i,
                word_idx_list=[decoder_init_input[batch_i]],
                decoder_state=decoder_init_state[:, batch_i, :].unsqueeze(1),  # [dec_layers, 1, decoder_size]
                memory_bank=memory_bank[batch_i].unsqueeze(0),  # [1, src_len, memory_bank_size]
                src_mask=src_mask[batch_i].unsqueeze(0),  # [1, src_len]
                src_oov=src_oov[batch_i].unsqueeze(0),  # [1, src_len]
                oov_list=oov_lists[batch_i],
                log_probs=[],
                score=0.0,
                context=context[batch_i].unsqueeze(0),  # [1, memory_bank_size]
                coverage=coverage[batch_i].unsqueeze(0) if self.coverage_attn else None,  # [1, src_len]
                attn_dist=[]
            )
            partial_sequences[batch_i].push(seq)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):
            # the total number of partial sequences of all the batches
            num_partial_sequences = sum([len(batch_seqs) for batch_seqs in partial_sequences])
            if num_partial_sequences == 0:
                # We have run out of partial candidates; often happens when beam_size is small
                break

            # flatten 2d sequences (batch_size, beam_size) into 1d batches (batch_size * beam_size) to feed model
            seq_idx2batch_idx, flattened_idx_map, y_t_flattened, decoder_state_flattened, memory_bank_flattened, src_mask_flattened, src_oov_flattened, context_flattened, coverage_flattened, oov_list_flattened = self.sequence_to_batch(partial_sequences)
            # seq_idx2batch_idx: [list of batch idx of seq in topN_heap_1, list of batch idx of seq in topN_heap_2, ...]
            # flattened_idx_map: [list of seq_id in topN_heap_1, list of seq_id in topN_heap_2, ...], seq_id increment from 0 at the first seq in first batch up to the end

            # Run one-step generation
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, h_t, context, attn_dist, _, coverage = \
                self.model.decoder(y_t_flattened, decoder_state_flattened, memory_bank_flattened, src_mask_flattened, context_flattened, max_num_oov, src_oov_flattened, coverage_flattened)

            log_decoder_dist = torch.log(decoder_dist + EPS)

            top_k_probs, top_k_word_indices = log_decoder_dist.data.topk(self.beam_size, dim=-1) # [flattened_batch, beam_size]

            '''
            # (batch_size * beam_size, 1, src_len) -> (batch_size * beam_size, src_len)
            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
            else:
                attn_weights = attn_weights.squeeze(1)
            '''

            '''
            # tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, hyp_seq_size, trg_hidden_dim), squeeze the first dim
            if isinstance(new_dec_hiddens, tuple):
                new_dec_hiddens1 = new_dec_hiddens[0].squeeze(0)
                new_dec_hiddens2 = new_dec_hiddens[1].squeeze(0)
                new_dec_hiddens = [(new_dec_hiddens1[i], new_dec_hiddens2[i]) for i in range(num_partial_sequences)]
            # convert to a list of hidden state for each beam, with dimension [hidden_dim]
            '''

            # For each partial sequence, push it to the TopN heap of the corresponding batch
            for batch_i in range(batch_size):
                num_new_hyp_in_batch = 0
                new_partial_sequences = TopN_heap(self.beam_size)

                for partial_idx, partial_seq in enumerate(partial_sequences[batch_i].extract()):
                    num_new_hyp = 0
                    flattened_seq_idx = flattened_idx_map[batch_i][partial_idx]

                    # check each new beam and decide to add to hypotheses or completed list
                    for beam_i in range(self.beam_size):
                        word_idx = top_k_word_indices[flattened_seq_idx][beam_i]

                        # score=0 means this is the first word <BOS>, empty the sentence
                        if partial_seq.score != 0:
                            new_word_idx_list = copy.copy(partial_seq.word_idx_list)
                        else:
                            new_word_idx_list = []
                        new_word_idx_list.append(word_idx)

                        new_partial_seq = Sequence(
                            batch_idx=partial_seq.batch_idx,
                            word_idx_list=new_word_idx_list,
                            decoder_state=h_t[:, flattened_seq_idx, :].unsqueeze(1),
                            memory_bank=partial_seq.memory_bank,
                            src_mask=partial_seq.src_mask,
                            src_oov=partial_seq.src_oov,
                            oov_list=partial_seq.oov_list,
                            log_probs=copy.copy(partial_seq.log_probs),
                            score=copy.copy(partial_seq.score),
                            context=context[flattened_seq_idx].unsqueeze(0),
                            coverage=coverage[flattened_seq_idx].unsqueeze(0) if self.coverage_attn else None,
                            attn_dist=copy.copy(partial_seq.attn_dist)
                        )

                        # we have generated self.beam_size new hypotheses for current hyp, stop generating
                        if num_new_hyp >= self.beam_size:
                            break

                        '''
                        if self.include_attn_dist:
                            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
                                new_partial_seq.attn_dist.append((attn_weights[0][flattened_seq_id], attn_weights[1][flattened_seq_id]))
                            else:
                                new_partial_seq.attn_dist.append(attn_weights[flattened_seq_id])
                        else:
                            new_partial_seq.attn_dist = None
                        '''

                        if self.include_attn_dist:
                            new_partial_seq.attn_dist.append(attn_dist[flattened_seq_idx].unsqueeze(0)) # [1, src_len]

                        new_partial_seq.log_probs.append(log_decoder_dist[flattened_seq_idx][beam_i])
                        new_partial_seq.score = new_partial_seq.score + log_decoder_dist[flattened_seq_idx][beam_i]

                        # if predict EOS, push it into complete_sequences
                        if word_idx == self.eos_idx:
                            if self.length_normalization_factor > 0:
                                L = self.length_normalization_const
                                length_penalty = (L + len(new_partial_seq.word_idx_list)) / (L + 1)
                                new_partial_seq.score /= length_penalty ** self.length_normalization_factor
                            complete_sequences[new_partial_seq.batch_idx].push(new_partial_seq)
                        else:
                            # print('Before pushing[%d]' % new_partial_sequences.size())
                            # print(sorted([s.score for s in new_partial_sequences._data]))
                            new_partial_sequences.push(new_partial_seq)
                            # print('After pushing[%d]' % new_partial_sequences.size())
                            # print(sorted([s.score for s in new_partial_sequences._data]))
                            num_new_hyp += 1
                            num_new_hyp_in_batch += 1

                    # print('Finished no.%d partial sequence' % partial_id)
                    # print('\t#(hypothese) = %d' % (len(new_partial_sequences)))
                    # print('\t#(completed) = %d' % (sum([len(c) for c in complete_sequences])))

                partial_sequences[batch_i] = new_partial_sequences

                #print('Batch=%d, \t#(hypothese) = %d, \t#(completed) = %d \t #(new_hyp_explored)=%d' % (batch_i, len(partial_sequences[batch_i]), len(complete_sequences[batch_i]), num_new_hyp_in_batch))
                '''
                # print-out for debug
                print('Source with OOV: \n\t %s' % ' '.join([str(w) for w in partial_seq.src_oov.cpu().data.numpy().tolist()]))
                print('OOV list: \n\t %s' % str(partial_seq.oov_list))

                for seq_id, seq in enumerate(new_partial_sequences._data):
                    print('%d, score=%.5f : %s' % (seq_id, seq.score, str(seq.sentence)))

                print('*' * 50)
                '''

            #print('Round=%d, \t#(batch) = %d, \t#(hypothese) = %d, \t#(completed) = %d' % (t, batch_size, sum([len(batch_heap) for batch_heap in partial_sequences]), sum([len(batch_heap) for batch_heap in complete_sequences])))

            # print('Round=%d' % (current_len))
            # print('\t#(hypothese) = %d' % (sum([len(batch_heap) for batch_heap in partial_sequences])))
            # for b_i in range(batch_size):
            #     print('\t\tbatch %d, #(hyp seq)=%d' % (b_i, len(partial_sequences[b_i])))
            # print('\t#(completed) = %d' % (sum([len(batch_heap) for batch_heap in complete_sequences])))
            # for b_i in range(batch_size):
            #     print('\t\tbatch %d, #(completed seq)=%d' % (b_i, len(complete_sequences[b_i])))

        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.

        # append all the partial_sequences to complete
        # [complete_sequences[s.batch_id] for s in partial_sequences]
        for batch_i in range(batch_size):
            if len(complete_sequences[batch_i]) == 0:
                complete_sequences[batch_i] = partial_sequences[batch_i].extract(sort=True)
            complete_sequences[batch_i] = complete_sequences[batch_i].extract(sort=True)

        return complete_sequences

