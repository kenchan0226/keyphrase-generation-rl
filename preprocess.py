import argparse
from collections import Counter
import torch
import pickle
import pykp.io
import config

def read_tokenized_src_file(path, concat_title=True):
    """
    read tokenized source text file and convert them to list of list of words
    :param path:
    :param concat_title: concatenate the words in title and content
    :return: data, a 2d list, each item in the list is a list of words of a src text, len(data) = num_lines
    """
    data = []
    with open(path) as f:
        for line in f:
            if concat_title:
                [title, context] = line.strip().split('<eos>')
                word_list = title.strip().split(' ') + context.strip().split(' ')
                data.append(word_list)
            else:
                raise ValueError('Not yet implement the function of separating title and context')
    return data

def read_tokenized_trg_file(path):
    """
    read tokenized target text file and convert them to list of list of words
    :param path:
    :return: data, a 3d list, each item in the list is a list of target, each target is a list of words.
    """
    data = []
    with open(path) as f:
        for line in f:
            trg_list = line.strip().split(';') # a list of target sequences
            trg_word_list = [trg.split(' ') for trg in trg_list]
            data.append(trg_word_list)
    return data

def build_vocab(tokenized_src_trg_pairs):
    token_freq_counter = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        # '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    return word2idx, idx2word, token_freq_counter

def main(opt):
    # Preprocess training data

    # Tokenize train_src and train_trg
    tokenized_train_src = read_tokenized_src_file(opt.train_src, concat_title=True)
    tokenized_train_trg = read_tokenized_trg_file(opt.train_trg)

    assert len(tokenized_train_src) == len(tokenized_train_trg), 'the number of records in source and target are not the same'

    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))
    # a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])

    del tokenized_train_src
    del tokenized_train_trg

    # build vocab from training src
    # build word2id, id2word, and vocab, where vocab is a counter
    # with special tokens, '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3
    # word2id, id2word are ordered by frequencies, includes all the tokens in the data
    # simply concatenate src and target when building vocab
    word2idx, idx2word, token_freq_counter = build_vocab(tokenized_train_pairs)

    # building preprocessed training set for one2one training mode
    train_one2one = pykp.io.build_dataset(tokenized_train_pairs, word2idx, idx2word, opt, mode='one2one')
    # a list of dict, with fields src, trg, src_oov, oov_dict, oov_list, etc.

    print("Dumping train one2one to disk: %s" % (opt.data_dir + '/train.one2one.pt'))
    torch.save(train_one2one, open(opt.data_dir + '/train.one2one.pt', 'wb'))
    len_train_one2one = len(train_one2one)
    del train_one2one
    # building preprocessed training set for one2many training mode
    train_one2many = pykp.io.build_dataset(tokenized_train_pairs, word2idx, idx2word, opt, mode='one2many')
    print("Dumping train one2many to disk: %s" % (opt.data_dir + '/train.one2many.pt'))
    torch.save(train_one2many, open(opt.data_dir + '/train.one2many.pt', 'wb'))
    len_train_one2many = len(train_one2many)
    del train_one2many

    # Preprocess validation data

    # Tokenize
    tokenized_valid_src = read_tokenized_src_file(opt.valid_src, concat_title=True)
    tokenized_valid_trg = read_tokenized_trg_file(opt.valid_trg)
    assert len(tokenized_valid_src) == len(
        tokenized_valid_trg), 'the number of records in source and target are not the same'

    tokenized_valid_pairs = list(zip(tokenized_valid_src, tokenized_valid_trg))
    del tokenized_valid_src
    del tokenized_valid_trg

    # building preprocessed validation set for one2one and one2many training mode
    valid_one2one = pykp.io.build_dataset(
        tokenized_valid_pairs, word2idx, idx2word, opt, mode='one2one', include_original=True)
    valid_one2many = pykp.io.build_dataset(
        tokenized_valid_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True)

    print("Dumping valid to disk: %s" % (opt.data_dir + '/valid.pt'))
    torch.save(valid_one2one, open(opt.data_dir+ '/valid.one2one.pt', 'wb'))
    torch.save(valid_one2many, open(opt.data_dir + '/valid.one2many.pt', 'wb'))

    # Preprocess test data
    tokenized_test_src = read_tokenized_src_file(opt.test_src, concat_title=True)
    tokenized_test_trg = read_tokenized_trg_file(opt.test_trg)
    assert len(tokenized_test_src) == len(
        tokenized_test_trg), 'the number of records in source and target are not the same'

    tokenized_test_pairs = list(zip(tokenized_test_src, tokenized_test_trg))
    del tokenized_test_src
    del tokenized_test_trg

    # building preprocessed test set for one2one and one2many training mode
    test_one2one = pykp.io.build_dataset(
        tokenized_test_pairs, word2idx, idx2word, opt, mode='one2one', include_original=True)
    test_one2many = pykp.io.build_dataset(
        tokenized_test_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True)

    print("Dumping test to disk: %s" % (opt.data_dir + '/valid.pt'))
    torch.save(test_one2one, open(opt.data_dir + '/test.one2one.pt', 'wb'))
    torch.save(test_one2many, open(opt.data_dir + '/test.one2many.pt', 'wb'))

    print("Dumping dict to disk: %s" % opt.data_dir + '/vocab.pt')
    torch.save([word2idx, idx2word, token_freq_counter],
               open(opt.data_dir + '/vocab.pt', 'wb'))

    print('#pairs of train_one2one  = %d' % len_train_one2one)
    print('#pairs of train_one2many = %d' % len_train_one2many)
    print('#pairs of valid_one2one  = %d' % len(valid_one2one))
    print('#pairs of valid_one2many = %d' % len(valid_one2many))
    print('#pairs of test_one2one   = %d' % len(test_one2one))
    print('#pairs of test_one2many  = %d' % len(test_one2many))

    print('Done!')

    '''
    special_tokens = ['<pad>']
    vocab = []
    vocab += special_tokens
    vocab += [w for w, n in token_freq_counter.items() if n > opt['word_count_threshold']]
    total_tokens = len(token_freq_counter)
    vocab_size = len(vocab) - len(special_tokens)
    OOV_words = total_tokens - vocab_size
    print('Vocab size: %d' % vocab_size)
    print('Number of OOV words: %d' % OOV_words)
    print('OOV percentage: %.2f' % OOV_words/total_tokens * 100 )
    '''
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The source files are tokenized and the tokens are separated by a space character.
    # The target sequences in the target files are separated by ';' character
    # data_dir should contains six files, train_src.txt, train_trg.txt, valid_src.txt, valid_trg.txt, test_src.txt, test_trg.txt

    parser.add_argument('-data_dir', required=True, help='The source file of the data')
    config.preprocess_opts(parser)
    #parser.add_argument('-vocab_size', default=50000, type=int, help='Max. number of words in vocab')
    #parser.add_argument('-max_unk_words', default=1000, type=int, help='Max. number of words in OOV vocab')
    opt = parser.parse_args()
    #opt = vars(args) # convert to dict
    opt.train_src = opt.data_dir + '/train_src.txt'
    opt.train_trg = opt.data_dir + '/train_trg.txt'
    opt.valid_src = opt.data_dir + '/valid_src.txt'
    opt.valid_trg = opt.data_dir + '/valid_trg.txt'
    opt.test_src = opt.data_dir + '/test_src.txt'
    opt.test_trg = opt.data_dir + '/test_trg.txt'
    main(opt)
