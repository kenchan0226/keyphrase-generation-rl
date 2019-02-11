import torch
import argparse
import config
import logging
import os
import json
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqModel
from torch.optim import Adam
import pykp

import train_ml
import train_rl

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
import time
import numpy as np
import random


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    if hasattr(opt, 'train_rl') and opt.train_rl:
        opt.exp += '.rl'

    if opt.one2many:
        opt.exp += '.one2many'

    if opt.one2many_mode == 1:
        opt.exp += '.cat'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if opt.use_target_encoder:
        opt.exp += '.target_encode'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    if opt.delimiter_type == 0:
        opt.delimiter_word = pykp.io.SEP_WORD
    else:
        opt.delimiter_word = pykp.io.EOS_WORD

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, opt.exp + '.initial.json'), 'w'))

    return opt


def init_optimizer_criterion(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
    '''
    if not opt.copy_attention:
        weight_mask = torch.ones(opt.vocab_size).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size)
    else:
        weight_mask = torch.ones(opt.vocab_size + opt.max_unk_words).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size + opt.max_unk_words)
    weight_mask[opt.word2id[pykp.IO.PAD_WORD]] = 0
    criterion = torch.nn.NLLLoss(weight=weight_mask)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    '''
    criterion = torch.nn.NLLLoss(ignore_index=opt.word2idx[pykp.io.PAD_WORD]).to(opt.device)

    if opt.train_ml:
        optimizer_ml = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        optimizer_ml = None

    if opt.train_rl:
        optimizer_rl = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate_rl)
    else:
        optimizer_rl = None

    return optimizer_ml, optimizer_rl, criterion


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.copy_attention:
        logging.info('Training a seq2seq model with copy mechanism')
    else:
        logging.info('Training a seq2seq model')
    model = Seq2SeqModel(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        # TODO: load the saved model and override the current one
    elif opt.train_rl and opt.pretrained_model != "":
        model.load_state_dict(torch.load(opt.pretrained_model))
        """
        pretrained_state_dict = torch.load(opt.pretrained_model)
        pretrained_state_dict_renamed = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith("encoder.rnn."):
                k = k.replace("encoder.rnn.", "encoder.encoder.rnn.", 1)
            pretrained_state_dict_renamed[k] = v
        model.load_state_dict(pretrained_state_dict_renamed)
        """
    return model.to(opt.device)


def main(opt):
    try:
        start_time = time.time()
        train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)
        start_time = time.time()
        model = init_model(opt)
        optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)
        if opt.train_ml:
            train_ml.train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt)
        else:
            train_rl.train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt)
        training_time = time_since(start_time)
        logging.info('Time for training: %.1f' % training_time)
    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    if opt.train_ml == opt.train_rl:
        raise ValueError("Either train with supervised learning or RL, but not both!")

    if not opt.one2many and opt.one2many_mode > 0:
        raise ValueError("You cannot choose one2many mode without the -one2many options.")

    if opt.one2many and opt.one2many_mode == 0:
        raise ValueError("If you choose one2many, you must specify the one2many mode.")

    if opt.one2many_mode == 1 and opt.num_predictions > 1:
        raise ValueError("If you set the one2many_mode to 1, the number of predictions should also be 1.")

    if not opt.one2many and opt.orthogonal_loss:
        raise ValueError("You can only use orthogonal loss in one2many mode.")

    if opt.mc_rollouts and opt.reward_shaping:
        raise ValueError("You cannot use monte-carlo rollout when using reward shaping")

    if opt.reward_shaping and opt.one2many_mode == 1:
        raise ValueError("You cannot use reward shapping when one2many mode=1")

    if opt.goal_vector_mode > 0 and not opt.separate_present_absent:
        raise ValueError("To use goal vector, you must use the option -separate_present_absent")

    if opt.topk != 'M' and opt.topk != 'G':
        opt.topk = int(opt.topk)

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
