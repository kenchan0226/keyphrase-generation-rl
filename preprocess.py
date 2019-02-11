import argparse
from collections import Counter
import torch
import pickle
import pykp.io
import config


def read_tokenized_src_file(path, remove_eos=True):
    """
    read tokenized source text file and convert them to list of list of words
    :param path:
    :param remove_eos: concatenate the words in title and content
    :return: data, a 2d list, each item in the list is a list of words of a src text, len(data) = num_lines
    """
    data = []
    with open(path) as f:
        for line in f:
            if remove_eos:
                title_and_context = line.strip().split('<eos>')
                if len(title_and_context) == 1:  # it only has context without title
                    [context] = title_and_context
                    word_list = context.strip().split(' ')
                elif len(title_and_context) == 2:
                    [title, context] = title_and_context
                    word_list = title.strip().split(' ') + context.strip().split(' ')
                else:
                    raise ValueError("The source text contains more than one title")
            else:
                word_list = line.strip().split(' ')
            data.append(word_list)
    return data


def read_tokenized_src_file(path, remove_eos=True, title_guided=False):
    """
    read tokenized source text file and convert them to list of list of words
    :param path:
    :param remove_eos: concatenate the words in title and content
    :return: data, a 2d list, each item in the list is a list of words of a src text, len(data) = num_lines
    """
    tokenized_train_src = []
    if title_guided:
        tokenized_train_title = []
    filtered_cnt = 0
    for line_idx, src_line in enumerate(open(path, 'r')):
        # process source line
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
            if title_guided:
                raise ValueError("The source text does not contains any title, so you cannot return title.")
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_eos:
                src_word_list = title_word_list + context_word_list
            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
        if title_guided:
            tokenized_train_title.append(title_word_list)

    if title_guided:
        return tokenized_train_src, tokenized_train_title
    else:
        return tokenized_train_src


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


def read_src_and_trg_files(src_file, trg_file, is_train, remove_eos=True, title_guided=False):
    tokenized_train_src = []
    tokenized_train_trg = []
    if title_guided:
        tokenized_train_title = []
    filtered_cnt = 0
    for line_idx, (src_line, trg_line) in enumerate(zip(open(src_file, 'r'), open(trg_file, 'r'))):
        # process source line
        if (len(src_line.strip()) == 0) and is_train:
            continue
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
            if title_guided:
                raise ValueError("The source text does not contains any title, so you cannot return title.")
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_eos:
                src_word_list = title_word_list + context_word_list
            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        # process target line
        trg_list = trg_line.strip().split(';')  # a list of target sequences
        trg_word_list = [trg.split(' ') for trg in trg_list]
        # If it is training data, ignore the line with source length > 400 or target length > 60
        if is_train:
            if len(src_word_list) > 400 or len(trg_word_list) > 14:
                filtered_cnt += 1
                continue
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
        tokenized_train_trg.append(trg_word_list)
        if title_guided:
            tokenized_train_title.append(title_word_list)

    assert len(tokenized_train_src) == len(
        tokenized_train_trg), 'the number of records in source and target are not the same'

    print("%d rows filtered" % filtered_cnt)

    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))

    if title_guided:
        return tokenized_train_pairs, tokenized_train_title
    else:
        return tokenized_train_pairs


def build_vocab(tokenized_src_trg_pairs, include_peos):
    token_freq_counter = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>', '<sep>']
    if include_peos:
        special_tokens.append('<peos>')
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
    """
    # Tokenize train_src and train_trg
    tokenized_train_src = read_tokenized_src_file(opt.train_src, remove_eos=opt.remove_eos)
    tokenized_train_trg = read_tokenized_trg_file(opt.train_trg)

    assert len(tokenized_train_src) == len(tokenized_train_trg), 'the number of records in source and target are not the same'

    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))
    # a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])

    del tokenized_train_src
    del tokenized_train_trg
    """
    title_guided = opt.title_guided

    # Tokenize train_src and train_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    if title_guided:
        tokenized_train_pairs, tokenized_train_title = read_src_and_trg_files(opt.train_src, opt.train_trg, is_train=True, remove_eos=opt.remove_eos, title_guided=True)
    else:
        tokenized_train_pairs = read_src_and_trg_files(opt.train_src, opt.train_trg, is_train=True, remove_eos=opt.remove_eos, title_guided=False)
        tokenized_train_title = None

    # build vocab from training src
    # build word2id, id2word, and vocab, where vocab is a counter
    # with special tokens, '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3
    # word2id, id2word are ordered by frequencies, includes all the tokens in the data
    # simply concatenate src and target when building vocab
    word2idx, idx2word, token_freq_counter = build_vocab(tokenized_train_pairs, opt.include_peos)

    # building preprocessed training set for one2one training mode
    train_one2one = pykp.io.build_dataset(tokenized_train_pairs, word2idx, idx2word, opt, mode='one2one', include_original=True, title_list=tokenized_train_title)
    # a list of dict, with fields src, trg, src_oov, oov_dict, oov_list, etc.

    print("Dumping train one2one to disk: %s" % (opt.data_dir + '/train.one2one.pt'))
    torch.save(train_one2one, open(opt.data_dir + '/train.one2one.pt', 'wb'))
    len_train_one2one = len(train_one2one)
    del train_one2one
    # building preprocessed training set for one2many training mode
    train_one2many = pykp.io.build_dataset(tokenized_train_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True, title_list=tokenized_train_title)
    print("Dumping train one2many to disk: %s" % (opt.data_dir + '/train.one2many.pt'))
    torch.save(train_one2many, open(opt.data_dir + '/train.one2many.pt', 'wb'))
    len_train_one2many = len(train_one2many)
    del train_one2many

    # Preprocess validation data
    """
    # Tokenize
    tokenized_valid_src = read_tokenized_src_file(opt.valid_src, remove_eos=opt.remove_eos)
    tokenized_valid_trg = read_tokenized_trg_file(opt.valid_trg)
    assert len(tokenized_valid_src) == len(
        tokenized_valid_trg), 'the number of records in source and target are not the same'

    tokenized_valid_pairs = list(zip(tokenized_valid_src, tokenized_valid_trg))
    del tokenized_valid_src
    del tokenized_valid_trg
    """
    # Tokenize valid_src and valid_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    if title_guided:
        tokenized_valid_pairs, tokenized_valid_title = read_src_and_trg_files(opt.valid_src, opt.valid_trg, is_train=False, remove_eos=opt.remove_eos, title_guided=True)
    else:
        tokenized_valid_pairs = read_src_and_trg_files(opt.valid_src, opt.valid_trg, is_train=False, remove_eos=opt.remove_eos, title_guided=False)
        tokenized_valid_title = None

    # building preprocessed validation set for one2one and one2many training mode
    valid_one2one = pykp.io.build_dataset(
        tokenized_valid_pairs, word2idx, idx2word, opt, mode='one2one', include_original=True, title_list=tokenized_valid_title)
    valid_one2many = pykp.io.build_dataset(
        tokenized_valid_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True, title_list=tokenized_valid_title)

    print("Dumping valid to disk: %s" % (opt.data_dir + '/valid.pt'))
    torch.save(valid_one2one, open(opt.data_dir+ '/valid.one2one.pt', 'wb'))
    torch.save(valid_one2many, open(opt.data_dir + '/valid.one2many.pt', 'wb'))

    # Preprocess test data
    """
    tokenized_test_src = read_tokenized_src_file(opt.test_src, remove_eos=opt.remove_eos)
    tokenized_test_trg = read_tokenized_trg_file(opt.test_trg)
    assert len(tokenized_test_src) == len(
        tokenized_test_trg), 'the number of records in source and target are not the same'

    tokenized_test_pairs = list(zip(tokenized_test_src, tokenized_test_trg))
    del tokenized_test_src
    del tokenized_test_trg
    """
    # Tokenize train_src and train_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    if title_guided:
        tokenized_test_pairs, tokenized_test_title = read_src_and_trg_files(opt.test_src, opt.test_trg, is_train=False, remove_eos=opt.remove_eos, title_guided=True)
    else:
        tokenized_test_pairs = read_src_and_trg_files(opt.test_src, opt.test_trg, is_train=False,
                                                      remove_eos=opt.remove_eos, title_guided=False)
        tokenized_test_title = None

    # building preprocessed test set for one2one and one2many training mode
    test_one2one = pykp.io.build_dataset(
        tokenized_test_pairs, word2idx, idx2word, opt, mode='one2one', include_original=True, title_list=tokenized_test_title)
    test_one2many = pykp.io.build_dataset(
        tokenized_test_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True, title_list=tokenized_test_title)

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
    parser.add_argument('-remove_eos', action="store_true", help='Remove the eos after the title')
    parser.add_argument('-include_peos', action="store_true", help='Include <peos> as a special token')
    parser.add_argument('-title_guided', action="store_true", help='Allow easy access to the title of the source text.')

    config.vocab_opts(parser)
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
