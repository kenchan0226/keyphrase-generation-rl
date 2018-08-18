import torch
import argparse
import config
import logging
import os
import json
from pykp.io import KeyphraseDataset
from pykp.dataloader import KeyphraseDataLoader
from torch.utils.data import DataLoader
from pykp.model import Seq2SeqModel
from torch.optim import Adam
import pykp
from beam_search import SequenceGenerator
from evaluate import evaluate_loss
import train_ml
from utils.statistics import Statistics
from utils.visualization import plot_train_valid_curve

EPS = 1e-8

def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    if hasattr(opt, 'train_rl') and opt.train_rl:
        opt.exp += '.rl'

    if hasattr(opt, 'copy_attention') and opt.copy_attention:
        opt.exp += '.copy'

    if hasattr(opt, 'coverage_attn') and opt.coverage_attn:
        opt.exp += 'coverage'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'


    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
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
    else:
        # TODO: dump the meta-model
        pass

    return model.to(opt.device)

def load_data_vocab(opt, load_train=True):
    # load vocab
    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2idx, idx2word, vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')

    # one2many data loader
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
        else:  # load one2many dataset
            train_one2many = torch.load(opt.data + '/train.one2many.pt', 'wb')
            train_one2many_dataset = KeyphraseDataset(train_one2many, word2idx=word2idx, idx2word=idx2word, type='one2many')
            train_loader = KeyphraseDataLoader(dataset=train_one2many_dataset, collate_fn=train_one2many_dataset.collate_fn_one2many, num_workers=opt.batch_workers, max_batch_example=1024, max_batch_pair=opt.batch_size, pin_memory=True, shuffle=True)
            logging.info('#(train data size: #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d, #(average examples/batch)=%.3f' % (len(train_loader.dataset), train_loader.one2one_number(), len(train_loader), train_loader.one2one_number() / len(train_loader)))
    else:
        train_loader = None


    if opt.train_ml:  # load one2one validation dataset
        valid_one2one = torch.load(opt.data + '/valid.one2one.pt', 'wb')
        valid_one2one_dataset = KeyphraseDataset(valid_one2one, word2idx=word2idx, idx2word=idx2word)
        valid_loader = DataLoader(dataset=valid_one2one_dataset,
                                          collate_fn=valid_one2one_dataset.collate_fn_one2one,
                                          num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                          shuffle=False)
        logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))
    else:
        valid_one2many = torch.load(opt.data + '/valid.one2many.pt', 'wb')
        # !important. As it takes too long to do beam search, thus reduce the size of validation and test datasets
        valid_one2many = valid_one2many[:2000]
        valid_one2many_dataset = KeyphraseDataset(valid_one2many, word2idx=word2idx, idx2word=idx2word, type='one2many',
                                                  include_original=True)
        valid_loader = KeyphraseDataLoader(dataset=valid_one2many_dataset, collate_fn=valid_one2many_dataset.collate_fn_one2many, num_workers=opt.batch_workers, max_batch_example=opt.beam_search_batch_example, max_batch_pair=opt.beam_search_batch_size, pin_memory=True, shuffle=False)
        logging.info(
            '#(valid data size: #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d, #(average examples/batch)=%.3f' % (
            len(valid_loader.dataset), valid_loader.one2one_number(), len(valid_loader),
            valid_loader.one2one_number() / len(valid_loader)))
    # assign vocab to opt
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab = vocab

    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return train_loader, valid_loader, word2idx, idx2word, vocab

def train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt):
    '''
    generator = SequenceGenerator(model,
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )
    '''
    logging.info('======================  Start Training  =========================')

    checkpoint_names = []
    # best_loss = sys.float_info.max # for normal training/testing loss (likelihood)
    best_loss = 0.0
    stop_increasing = 0
    total_train_statistics = Statistics()
    report_train_statistics = Statistics()
    report_train_ppl = []
    report_valid_ppl = []

    total_batch = -1
    early_stop_flag = False

    best_valid_loss = float('inf')
    num_stop_dropping = 0

    '''
    if opt.train_rl:
        reward_cache = RewardCache(2000)
    '''

    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    for epoch in range(opt.start_epoch, opt.epochs):
        if early_stop_flag:
            break

        # TODO: progress bar
        #progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,total_examples=len(train_data_loader.dataset.examples))

        for batch_i, batch in enumerate(train_data_loader):
            model.train()
            total_batch += 1
            report_loss = []

            # Training
            if opt.train_ml:
                batch_loss_stat, decoder_dist = train_ml.train_one_batch(batch, model, optimizer_ml, opt)
                report_train_statistics.update(batch_loss_stat)
                total_train_statistics.update(batch_loss_stat)
                #report_loss.append(('train_ml_loss', loss_ml))
                #report_loss.append(('PPL', loss_ml))

                # Brief report
                '''
                if batch_i % opt.report_every == 0:
                    brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt)
                '''
            else:
                # TODO: traing_rl
                '''
                if epoch >= opt.rl_start_epoch:
                    loss_rl = train_rl(one2many_batch, model, optimizer_rl, generator, opt, reward_cache)
                else:
                    loss_rl = 0.0
                train_rl_losses.append(loss_rl)
                report_loss.append(('train_rl_loss', loss_rl))
                '''
                pass

            #progbar.update(epoch, batch_i, report_loss)

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            if opt.train_ml:
                if (opt.run_valid_every == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.run_valid_every > -1 and total_batch > 1 and total_batch % opt.run_valid_every == 0):
                    # test the model on the validation dataset for one epoch
                    eval_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                    current_eval_loss = eval_loss_stat.xent()

                    if current_eval_loss < best_valid_loss:
                        best_valid_loss = current_eval_loss
                        num_stop_dropping = 0
                    else:
                        num_stop_dropping += 1
                        # decay the learning rate by a factor
                        for i, param_group in enumerate(optimizer_ml.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr

                    current_train_ppl = report_train_statistics.ppl()
                    current_eval_ppl = eval_loss_stat.ppl()
                    logging.info('Average training perplexity: %.3f; Average validation perplexity: %.3f; over %d iterations' % (current_train_ppl, current_eval_loss, opt.run_valid_every))

                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_eval_ppl)

                    if num_stop_dropping >= opt.early_stop_tolerance:
                        logging.info('Have not increased for %d epochs, early stop training' % num_stop_dropping)
                        early_stop_flag = True
                        break

                    report_train_statistics.reset()

    train_valid_curve_path = opt.exp_path + '/train_valid_ppl.pdf'
    plot_train_valid_curve(report_train_ppl, report_valid_ppl, opt.run_valid_every, train_valid_curve_path)
    logging.info('Overall average training loss: %.3f, ppl: %.3f' % (total_train_statistics.xent(), total_train_statistics.ppl()))

def main(opt):
    try:
        train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_vocab(opt)
        model = init_model(opt)
        optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)
        train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt)
    except Exception as e:
        logging.exception("message")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.train_ml == opt.train_rl:
        raise ValueError("Either train with supervised learning or RL, but not both!")

    logging = config.init_logging(logger_name=None, log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)

