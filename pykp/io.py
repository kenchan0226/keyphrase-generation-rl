# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""
import codecs
import inspect
import itertools
import json
import re
import traceback
from collections import Counter
from collections import defaultdict
import numpy as np
import sys

#import torchtext
import torch
import torch.utils.data

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'
PEOS_WORD = '<peos>'


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, type='one2many', delimiter_type=0, load_train=True, remove_src_eos=False, title_guided=False):
        # keys of matter. `src_oov_map` is for mapping pointed word to dict, `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch
        assert type in ['one2one', 'one2many']
        if type == 'one2one':
            keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list']
        elif type == 'one2many':
            keys = ['src', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'trg', 'trg_copy']

        if title_guided:
            keys += ['title', 'title_oov']

        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
                '''
                if type == 'one2one':
                    filtered_example['oov_number'] = len(filtered_example['oov_list'])
                elif type == 'one2many':
                    # TODO: check the oov_number field in one2many example
                    filtered_example['oov_number'] = [len(oov) for oov in filtered_example['oov_list']]
                '''

            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.pad_idx = word2idx[PAD_WORD]
        self.type = type
        if delimiter_type == 0:
            self.delimiter = self.word2idx[SEP_WORD]
        else:
            self.delimiter = self.word2idx[EOS_WORD]
        self.load_train = load_train
        self.remove_src_eos = remove_src_eos
        self.title_guided = title_guided

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.pad_idx)
        input_mask = input_mask.type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        assert self.type == 'one2one', 'The type of dataset should be one2one.'
        if self.remove_src_eos:
            # source with oov words replaced by <unk>
            src = [b['src'] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] for b in batches]
        else:
            # source with oov words replaced by <unk>
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        if self.title_guided:
            title = [b['title'] for b in batches]
            title_oov = [b['title_oov'] for b in batches]
        else:
            title, title_oov, title_lens, title_mask = None, None, None, None

        """
        src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
        # src = [[self.word2idx[BOS_WORD]] + b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
        # extended src (unk words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]
        # src_oov = [[self.word2idx[BOS_WORD]] + b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]
        """

        # target_input: input to decoder, ends with <eos> and oovs are replaced with <unk>
        trg = [b['trg'] + [self.word2idx[EOS_WORD]] for b in batches]

        # target for copy model, ends with <eos>, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_oov = [b['trg_copy'] + [self.word2idx[EOS_WORD]] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        if self.title_guided:
            seq_pairs = sorted(zip(src, trg, trg_oov, src_oov, oov_lists, title, title_oov), key=lambda p: len(p[0]), reverse=True)
            src, trg, trg_oov, src_oov, oov_lists, title, title_oov = zip(*seq_pairs)
            title, title_lens, title_mask = self._pad(title)
            title_oov, _, _ = self._pad(title_oov)
        else:
            seq_pairs = sorted(zip(src, trg, trg_oov, src_oov, oov_lists), key=lambda p: len(p[0]), reverse=True)
            src, trg, trg_oov, src_oov, oov_lists = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        trg, trg_lens, trg_mask = self._pad(trg)

        #trg_target, _, _ = self._pad(trg_target)
        trg_oov, _, _ = self._pad(trg_oov)
        src_oov, _, _ = self._pad(src_oov)

        return src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, title, title_oov, title_lens, title_mask

    def collate_fn_one2many(self, batches):
        assert self.type == 'one2many', 'The type of dataset should be one2many.'
        if self.remove_src_eos:
            # source with oov words replaced by <unk>
            src = [b['src'] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] for b in batches]
        else:
            # source with oov words replaced by <unk>
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        if self.title_guided:
            title = [b['title'] for b in batches]
            title_oov = [b['title_oov'] for b in batches]
        else:
            title, title_oov, title_lens, title_mask = None, None, None, None

        batch_size = len(src)

        # trg: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oov replaced by UNK
        # trg_oov: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b['trg_copy'])):
                    # b['trg'] contains a list of targets (keyphrase), each target is a list of indices, 2d list of idx
                #for trg_idx, a in enumerate(zip(b['trg'], b['trg_copy'])):
                    #trg_phase, trg_phase_oov are list of idx
                    if trg_phase[0] == self.word2idx[PEOS_WORD]:
                        if trg_idx == 0:
                            trg_concat += trg_phase
                            trg_oov_concat += trg_phase_oov
                        else:
                            trg_concat[-1] = trg_phase[0]
                            trg_oov_concat[-1] = trg_phase_oov[0]
                            if trg_idx == trg_size - 1:
                                trg_concat.append(self.word2idx[EOS_WORD])
                                trg_oov_concat.append(self.word2idx[EOS_WORD])
                    else:
                        if trg_idx == trg_size - 1:  # if this is the last keyphrase, end with <eos>
                            trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                            trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                        else:
                            trg_concat += trg_phase + [self.delimiter]  # trg_concat = [target_1] + [delimiter] + [target_2] + [delimiter] + ...
                            trg_oov_concat += trg_phase_oov + [self.delimiter]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)
        else:
            trg, trg_oov = None, None
        #trg = [[t + [self.word2idx[EOS_WORD]] for t in b['trg']] for b in batches]
        #trg_oov = [[t + [self.word2idx[EOS_WORD]] for t in b['trg_copy']] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]

        # b['src_str'] is a word_list for source text, b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        original_indices = list(range(batch_size))

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        if self.load_train:
            if self.title_guided:
                seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices, title, title_oov),
                                   key=lambda p: len(p[0]), reverse=True)
                src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices, title, title_oov = zip(*seq_pairs)
            else:
                seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices),
                                   key=lambda p: len(p[0]), reverse=True)
                src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices = zip(*seq_pairs)
        else:
            if self.title_guided:
                seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, original_indices, title, title_oov),
                                   key=lambda p: len(p[0]), reverse=True)
                src, src_oov, oov_lists, src_str, trg_str, original_indices, title, title_oov = zip(*seq_pairs)
            else:
                seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, original_indices),
                                   key=lambda p: len(p[0]), reverse=True)
                src, src_oov, oov_lists, src_str, trg_str, original_indices = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)
        if self.load_train:
            trg, trg_lens, trg_mask = self._pad(trg)
            trg_oov, _, _ = self._pad(trg_oov)
        else:
            trg_lens, trg_mask = None, None

        if self.title_guided:
            title, title_lens, title_mask = self._pad(title)
            title_oov, _, _ = self._pad(title_oov)

        return src, src_lens, src_mask, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices, title, title_oov, title_lens, title_mask

    def collate_fn_one2many_hier(self, batches):
        assert self.type == 'one2many', 'The type of dataset should be one2many.'
        # source with oov words replaced by <unk>
        src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        batch_size = len(src)

        # trg: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oov replaced by UNK
        # trg_oov: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b[
                    'trg_copy'])):  # b['trg'] contains a list of targets, each target is a list of indices
                    # for trg_idx, a in enumerate(zip(b['trg'], b['trg_copy'])):
                    # trg_phase, trg_phase_oov = a
                    if trg_idx == trg_size - 1:  # if this is the last keyphrase, end with <eos>
                        trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                    else:
                        trg_concat += trg_phase + [
                            self.delimiter]  # trg_concat = [target_1] + [delimiter] + [target_2] + [delimiter] + ...
                        trg_oov_concat += trg_phase_oov + [self.delimiter]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)
        else:
            trg, trg_oov = None, None
        # trg = [[t + [self.word2idx[EOS_WORD]] for t in b['trg']] for b in batches]
        # trg_oov = [[t + [self.word2idx[EOS_WORD]] for t in b['trg_copy']] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]

        # b['src_str'] is a word_list for source text, b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        original_indices = list(range(batch_size))

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        if self.load_train:
            seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices),
                               key=lambda p: len(p[0]), reverse=True)
            src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices = zip(*seq_pairs)
        else:
            seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, original_indices),
                               key=lambda p: len(p[0]), reverse=True)
            src, src_oov, oov_lists, src_str, trg_str, original_indices = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)
        if self.load_train:
            trg, trg_lens, trg_mask = self._pad(trg)
            trg_oov, _, _ = self._pad(trg_oov)
        else:
            trg_lens, trg_mask = None, None

        return src, src_lens, src_mask, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices

'''
class KeyphraseDatasetTorchText(torchtext.data.Dataset):
    @staticmethod
    def sort_key(ex):
        return torchtext.data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, raw_examples, fields, **kwargs):
        """Create a KeyphraseDataset given paths and fields. Modified from the TranslationDataset
        Arguments:
            examples: The list of raw examples in the dataset, each example is a tuple of two lists (src_tokens, trg_tokens)
            fields: A tuple containing the fields that will be used for source and target data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        for (src_tokens, trg_tokens) in raw_examples:
            examples.append(torchtext.data.Example.fromlist(
                [src_tokens, trg_tokens], fields))

        super(KeyphraseDatasetTorchText, self).__init__(examples, fields, **kwargs)
'''

def load_json_data(path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';'):
    '''
    To load keyphrase data from file, generate src by concatenating the contents in src_fields
    Input file should be json format, one document per line
    return pairs of (src_str, [trg_str_1, trg_str_2 ... trg_str_m])
    default data is 'kp20k'
    :param train_path:
    :param name:
    :param src_fields:
    :param trg_fields:
    :param trg_delimiter:
    :return:
    '''
    src_trgs_pairs = []
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(corpus_file):
            # if(idx == 20000):
            #     break
            # print(line)
            json_ = json.loads(line)

            trg_strs = []
            src_str = '.'.join([json_[f] for f in src_fields])
            [trg_strs.extend(re.split(trg_delimiter, json_[f])) for f in trg_fields]
            src_trgs_pairs.append((src_str, trg_strs))

    return src_trgs_pairs


def copyseq_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))

    # replace the digit terms with <digit>
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens


def tokenize_filter_data(
        src_trgs_pairs, tokenize, opt, valid_check=False):
    '''
    tokenize and truncate data, filter examples that exceed the length limit
    :param src_trgs_pairs:
    :param tokenize:
    :param src_seq_length:
    :param trg_seq_length:
    :param src_seq_length_trunc:
    :param trg_seq_length_trunc:
    :return:
    '''
    return_pairs = []
    for idx, (src, trgs) in enumerate(src_trgs_pairs):
        src_filter_flag = False

        src = src.lower() if opt.lower else src
        src_tokens = tokenize(src)
        if opt.src_seq_length_trunc and len(src) > opt.src_seq_length_trunc:
            src_tokens = src_tokens[:opt.src_seq_length_trunc]

        # FILTER 3.1: if length of src exceeds limit, discard
        if opt.max_src_seq_length and len(src_tokens) > opt.max_src_seq_length:
            src_filter_flag = True
        if opt.min_src_seq_length and len(src_tokens) < opt.min_src_seq_length:
            src_filter_flag = True

        if valid_check and src_filter_flag:
            continue

        trgs_tokens = []
        for trg in trgs:
            trg_filter_flag = False
            trg = trg.lower() if src.lower else trg

            # FILTER 1: remove all the abbreviations/acronyms in parentheses in keyphrases
            trg = re.sub(r'\(.*?\)', '', trg)
            trg = re.sub(r'\[.*?\]', '', trg)
            trg = re.sub(r'\{.*?\}', '', trg)

            # FILTER 2: ingore all the phrases that contains strange punctuations, very DIRTY data!
            puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', trg)

            trg_tokens = tokenize(trg)

            if len(puncts) > 0:
                print('-' * 50)
                print('Find punctuations in keyword: %s' % trg)
                print('- tokens: %s' % str(trg_tokens))
                continue

            # FILTER 3.2: if length of trg exceeds limit, discard
            if opt.trg_seq_length_trunc and len(trg) > opt.trg_seq_length_trunc:
                trg_tokens = trg_tokens[:src.trg_seq_length_trunc]
            if opt.max_trg_seq_length and len(trg_tokens) > opt.max_trg_seq_length:
                trg_filter_flag = True
            if opt.min_trg_seq_length and len(trg_tokens) < opt.min_trg_seq_length:
                trg_filter_flag = True

            filtered_by_heuristic_rule = False

            # FILTER 4: check the quality of long keyphrases (>5 words) with a heuristic rule
            if len(trg_tokens) > 5:
                trg_set = set(trg_tokens)
                if len(trg_set) * 2 < len(trg_tokens):
                    filtered_by_heuristic_rule = True

            if valid_check and (trg_filter_flag or filtered_by_heuristic_rule):
                print('*' * 50)
                if filtered_by_heuristic_rule:
                    print('INVALID by heuristic_rule')
                else:
                    print('VALID by heuristic_rule')
                print('length of src/trg exceeds limit: len(src)=%d, len(trg)=%d' % (len(src_tokens), len(trg_tokens)))
                print('src: %s' % str(src))
                print('trg: %s' % str(trg))
                print('*' * 50)
                continue

            # FILTER 5: filter keywords like primary 75v05;secondary 76m10;65n30
            if (len(trg_tokens) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d', trg_tokens[0].strip())) or (len(trg_tokens) > 1 and re.match(r'\d\d\w\d\d', trg_tokens[1].strip())):
                print('Find dirty keyword of type \d\d[a-z]\d\d: %s' % trg)
                continue

            trgs_tokens.append(trg_tokens)

        return_pairs.append((src_tokens, trgs_tokens))

        if idx % 2000 == 0:
            print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
            print(src)
            print(src_tokens)
            print(trgs)
            print(trgs_tokens)

    return return_pairs


def build_interactive_predict_dataset(tokenized_src, word2idx, idx2word, opt, title_list=None):
    # build a dummy trg list, and then combine it with src, and pass it to the build_dataset method
    num_lines = len(tokenized_src)
    tokenized_trg = [['.']] * num_lines  # create a dummy tokenized_trg
    tokenized_src_trg_pairs = list(zip(tokenized_src, tokenized_trg))
    return build_dataset(tokenized_src_trg_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True, title_list=title_list)


def build_dataset(src_trgs_pairs, word2idx, idx2word, opt, mode='one2one', include_original=False, title_list=None):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    return_examples = []
    oov_target = 0
    max_oov_len = 0
    max_oov_sent = ''
    if title_list != None:
        assert len(title_list) == len(src_trgs_pairs)

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        # if w is not seen in training data vocab (word2idx, size could be larger than opt.vocab_size), replace with <unk>
        #src_all = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in source]
        # if w's id is larger than opt.vocab_size, replace with <unk>
        src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in source]

        if title_list is not None:
            title_word_list = title_list[idx]
            #title_all = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in title_word_list]
            title = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in title_word_list]

        # create a local vocab for the current source text. If there're V words in the vocab of this string, len(itos)=V+2 (including <unk> and <pad>), len(stoi)=V+1 (including <pad>)
        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)
        examples = []  # for one-to-many

        for target in targets:
            example = {}

            if include_original:
                example['src_str'] = source
                example['trg_str'] = target

            example['src'] = src
            # example['src_input'] = [word2idx[BOS_WORD]] + src + [word2idx[EOS_WORD]] # target input, requires BOS at the beginning
            # example['src_all']   = src_all

            if title_list is not None:
                example['title'] = title

            trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in target]
            example['trg'] = trg
            # example['trg_input']   = [word2idx[BOS_WORD]] + trg + [word2idx[EOS_WORD]] # target input, requires BOS at the beginning
            # example['trg_all']   = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in target]
            # example['trg_loss']  = example['trg'] + [word2idx[EOS_WORD]] # target for loss computation, ignore BOS

            example['src_oov'] = src_oov
            example['oov_dict'] = oov_dict
            example['oov_list'] = oov_list
            if len(oov_list) > max_oov_len:
                max_oov_len = len(oov_list)
                max_oov_sent = source

            # oov words are replaced with new index
            trg_copy = []
            for w in target:
                if w in word2idx and word2idx[w] < opt.vocab_size:
                    trg_copy.append(word2idx[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2idx[UNK_WORD])
            example['trg_copy'] = trg_copy

            if title_list is not None:
                title_oov = []
                for w in title_word_list:
                    if w in word2idx and word2idx[w] < opt.vocab_size:
                        title_oov.append(word2idx[w])
                    elif w in oov_dict:
                        title_oov.append(oov_dict[w])
                    else:
                        title_oov.append(word2idx[UNK_WORD])
                example['title_oov'] = title_oov

            # example['trg_copy_input'] = [word2idx[BOS_WORD]] + trg_copy + [word2idx[EOS_WORD]] # target input, requires BOS at the beginning
            # example['trg_copy_loss']  = example['trg_copy'] + [word2idx[EOS_WORD]] # target for loss computation, ignore BOS

            # example['copy_martix'] = copy_martix(source, target)
            # C = [0 if w not in source else source.index(w) + opt.vocab_size for w in target]
            # example["copy_index"] = C
            # A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            # B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
            # C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

            if any([w >= opt.vocab_size for w in trg_copy]):
                oov_target += 1

            if idx % 100000 == 0:
                print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
                print('source    \n\t\t[len=%d]: %s' % (len(source), source))
                print('target    \n\t\t[len=%d]: %s' % (len(target), target))
                # print('src_all   \n\t\t[len=%d]: %s' % (len(example['src_all']), example['src_all']))
                # print('trg_all   \n\t\t[len=%d]: %s' % (len(example['trg_all']), example['trg_all']))
                print('src       \n\t\t[len=%d]: %s' % (len(example['src']), example['src']))
                # print('src_input \n\t\t[len=%d]: %s' % (len(example['src_input']), example['src_input']))
                print('trg       \n\t\t[len=%d]: %s' % (len(example['trg']), example['trg']))
                # print('trg_input \n\t\t[len=%d]: %s' % (len(example['trg_input']), example['trg_input']))

                print('src_oov   \n\t\t[len=%d]: %s' % (len(src_oov), src_oov))

                print('oov_dict         \n\t\t[len=%d]: %s' % (len(oov_dict), oov_dict))
                print('oov_list         \n\t\t[len=%d]: %s' % (len(oov_list), oov_list))
                if len(oov_dict) > 0:
                    print('Find OOV in source')

                print('trg_copy         \n\t\t[len=%d]: %s' % (len(trg_copy), trg_copy))
                # print('trg_copy_input   \n\t\t[len=%d]: %s' % (len(example["trg_copy_input"]), example["trg_copy_input"]))

                if any([w >= opt.vocab_size for w in trg_copy]):
                    print('Find OOV in target')

                # print('copy_martix      \n\t\t[len=%d]: %s' % (len(example["copy_martix"]), example["copy_martix"]))
                # print('copy_index  \n\t\t[len=%d]: %s' % (len(example["copy_index"]), example["copy_index"]))

            if mode == 'one2one':
                return_examples.append(example)
                '''
                For debug
                if len(oov_list) > 0:
                    print("Found oov")
                '''
            else:
                examples.append(example)

        if mode == 'one2many' and len(examples) > 0:
            o2m_example = {}
            keys = examples[0].keys()
            for key in keys:
                if key.startswith('src') or key.startswith('oov') or key.startswith('title'):
                    o2m_example[key] = examples[0][key]
                else:
                    o2m_example[key] = [e[key] for e in examples]
            if include_original:
                assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
            else:
                assert len(o2m_example['src']) == len(o2m_example['src_oov'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])
            if title_list is not None:
                assert len(o2m_example['title']) == len(o2m_example['title_oov'])

            return_examples.append(o2m_example)

    print('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(return_examples)))
    print('Find max_oov_len = %d' % (max_oov_len))
    print('max_oov sentence: %s' % str(max_oov_sent))

    return return_examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    WARNING: if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    Args:
        source_words: list of words (strings)
        word2idx: vocab word2idx
        vocab_size: the maximum acceptable index of word in vocab
    Returns:
        ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
    """
    src_oov = []
    oov_dict = {}
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx[UNK_WORD]
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x:x[1])]
    return src_oov, oov_dict, oov_list


def copy_martix(source, target):
    '''
    For reproduce Gu's method
    return the copy matrix, size = [nb_sample, max_len_source, max_len_target]
    cc_matrix[i][j]=1 if i-th word in target matches the i-th word in source
    '''
    cc = np.zeros((len(target), len(source)), dtype='float32')
    for i in range(len(target)):  # go over each word in target (all target have same length after padding)
        for j in range(len(source)):  # go over each word in source
            if source[j] == target[i]:  # if word match, set cc[k][j][i] = 1. Don't count non-word(source[k, i]=0)
                cc[i][j] = 1.
    return cc

'''
def build_vocab(tokenized_src_trgs_pairs, opt):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for src_tokens, trgs_tokens in tokenized_src_trgs_pairs:
        tokens = src_tokens + list(itertools.chain(*trgs_tokens))
        for token in tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1

    # Discard start, end, pad and unk tokens if already present
    if '<bos>' in vocab:
        del vocab['<bos>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '<eos>' in vocab:
        del vocab['<eos>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    word2idx = {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        '<unk>': 3,
    }

    idx2word = {
        0: '<pad>',
        1: '<bos>',
        2: '<eos>',
        3: '<unk>',
    }

    sorted_word2id = sorted(
        vocab.items(),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id]

    for ind, word in enumerate(sorted_words):
        word2idx[word] = ind + 4

    for ind, word in enumerate(sorted_words):
        idx2word[ind + 4] = word

    return word2idx, idx2word, vocab
'''

'''
def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab
'''
