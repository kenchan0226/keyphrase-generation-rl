import torch
import logging
from pykp.io import KeyphraseDataset
from torch.utils.data import DataLoader

def load_data_and_vocab(opt, load_train=True):
    # load vocab
    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2idx, idx2word, vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')

    # assign vocab to opt
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab = vocab
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)

    if load_train:  # load training dataset
        if opt.train_ml: # load one2one dataset
            train_one2one = torch.load(opt.data + '/train.one2one.pt', 'wb')
            train_one2one_dataset = KeyphraseDataset(train_one2one, word2idx=word2idx, idx2word=idx2word, type='one2one')
            train_loader = DataLoader(dataset=train_one2one_dataset,
                                              collate_fn=train_one2one_dataset.collate_fn_one2one,
                                              num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                              shuffle=True)
            logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

            valid_one2one = torch.load(opt.data + '/valid.one2one.pt', 'wb')
            valid_one2one_dataset = KeyphraseDataset(valid_one2one, word2idx=word2idx, idx2word=idx2word,
                                                     type='one2one')
            valid_loader = DataLoader(dataset=valid_one2one_dataset,
                                      collate_fn=valid_one2one_dataset.collate_fn_one2one,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=False)
            logging.info('#(valid data size: #(batch)=%d' % (len(train_loader)))


        else:  # load one2many dataset
            train_one2many = torch.load(opt.data + '/train.one2many.pt', 'wb')
            train_one2many_dataset = KeyphraseDataset(train_one2many, word2idx=word2idx, idx2word=idx2word, type='one2many')
            train_loader = DataLoader(dataset=train_one2many_dataset,
                                      collate_fn=train_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=True)
            logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

            valid_one2many = torch.load(opt.data + '/valid.one2many.pt', 'wb')
            #valid_one2many = valid_one2many[:2000]
            valid_one2many_dataset = KeyphraseDataset(valid_one2many, word2idx=word2idx, idx2word=idx2word,
                                                      type='one2many')
            valid_loader = DataLoader(dataset=valid_one2many_dataset,
                                      collate_fn=valid_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=False)
            logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, valid_loader, word2idx, idx2word, vocab
    else:
        test_one2many = torch.load(opt.data + '/test.one2many.pt', 'wb')
        test_one2many_dataset = KeyphraseDataset(test_one2many, word2idx=word2idx, idx2word=idx2word,
                                                      type='one2many')
        test_loader = DataLoader(dataset=test_one2many_dataset,
                                      collate_fn=test_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

        return test_loader, word2idx, idx2word, vocab
