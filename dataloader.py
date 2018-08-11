import numpy as np
import random
import torch

class DataLoader():
    def __init__(self, data_dir, type, word2idx, device):
        self.word2idx = word2idx
        assert type in ['train', 'valid', 'test']
        self.type = type
        self.device
        '''
        self.src_data = open(src_path).readlines()
        self.tgt_data = open(tgt_path).readlines()
        if shuffle:
            random_idx = np.random.permutation(len(src_data))
            src_data = [src_data[idx] for idx in random_idx]
            tgt_data = [tgt_data[idx] for idx in random_idx]
        self.current_idx = 0
        '''

        src_data = []  # list of list of word idx, [N, seq_len]
        tgt_data = []  # list of list of list of word idx, [N, M_i, seq_len]

        src_path = data_dir + '/' + type + '_src.txt'
        tgt_path = data_dir + '/' + type + '_tgt.txt'

        with open(src_path) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index in wod2idx
                s = [word2idx[token] for token in sentence.strip().split(' ')]
                src_data.append(s)

        with open(tgt_path) as f:
            for sentences in f.read().splitlines():
                sentence_list = sentences.split(';')
                for sentence in sentence_list:
                    s = [word2idx[token] for token in sentence.strip().split(' ')]
                l = [tag_map[label] for label in sentence.split(' ')]
                train_labels.append(l)

        assert len(src_data) == len(tgt_data), 'The number of lines of source and target do not match.'

        self.src_data = src_data
        self.tgt_data = tgt_data
        self.data_size = len(src_data)

    def __len__(self):
        return len(self.tgt_data)

    def data_iterator(self, batch_size, shuffle=False):
        permutation = list(range(self.data_size))
        if shuffle:
            random.seed(230)
            permutation = random.shuffle(permutation)
        # one pass over the data, iterate through every batch i
        for i in range((self.data_size + 1) // batch_size):
            # Load a batch of data according to the index in permutation
            src_batch_list = [ self.src_data[idx] for idx in permutation[i * batch_size: (i+1) * batch_size] ]
            # target_batch

            # Compute the maximum seq_len in the src batch
            max_src_seq_len = max([len(s) for s in src_batch_list])

            # prepare a numpy array with the data, initializing the data with pad_idx
            src_batch = pad_idx * np.ones((len(src_batch_list), max_src_seq_len))
            # pad target?

            # copy the data to the numpy array
            for j in range(len(src_batch_list)):
                seq_len = len(src_batch_list[j])
                src_batch[j][:seq_len] = src_batch_list[j]
                # target?

            src_batch = torch.Tensor(src_batch, dtype=torch.long, device=self.device, requires_grad=True)

        yield src_batch, tgt_batch
