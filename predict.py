import torch
from beam_search import SequenceGenerator
import logging
import config
from pykp.io import KeyphraseDataset
from torch.utils.data import DataLoader
import time
from utils.time_log import time_since
from evaluate import evaluate_beam_search
import pykp.io
import sys
import argparse

def load_data_vocab(opt):
    # load vocab
    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2idx, idx2word, vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')

    # constructor data loader
    logging.info("Loading test data from '%s'" % opt.data)

    test_one2many = torch.load(opt.data + '/test.one2many.pt', 'wb')
    test_one2many_dataset = KeyphraseDataset(test_one2many, word2idx=word2idx, idx2word=idx2word, type='one2many')
    test_loader = DataLoader(dataset=test_one2many_dataset,
                              collate_fn=test_one2many_dataset.collate_fn_one2many,
                              num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                              shuffle=False)
    logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

    return test_loader, word2idx, idx2word, vocab

def main(opt):
    logging = config.init_logging(logger_name='predict', log_file=opt.exp_path + '/output.log')
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    try:
        start_time = time.time()
        test_data_loader, word2idx, idx2word, vocab = load_data_vocab(opt)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)
        # TODO: load pretrained model
        model = None
        generator = SequenceGenerator(model,
                                      eos_idx=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length)
        evaluate_beam_search(generator, test_data_loader, opt)
        total_testing_time = time_since(start_time)
        logging.info('Time for a complete testing: %.1f' % total_testing_time)

    except Exception as e:
        logging.exception("message")
    return

    pass

if __name__=='__main__':
    # TODO: parse arguments

    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.predict_opts(parser)
    opt = parser.parse_args()

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    print(opt.gpuid)
    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    opt.exp = 'predict.' + opt.exp
    if hasattr(opt, 'copy_model') and opt.copy_model:
        opt.exp += '.copy'

    if hasattr(opt, 'bidirectional'):
        if opt.bidirectional:
            opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    main(opt)
