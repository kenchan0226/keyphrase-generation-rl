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
from sequence_generator import SequenceGenerator
from evaluate import evaluate_loss
import train_ml
#import train_rl
from utils.statistics import Statistics
from utils.report import export_train_and_valid_results
from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
import time
import math
import sys

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

    #if hasattr(opt, 'one2many') and opt.one2many:
    #    opt.exp += '.one2many'

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
    else:
        pass

    return model.to(opt.device)


def train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt):
    '''
    generator = SequenceGenerator(model,
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )
    '''
    logging.info('======================  Start Training  =========================')

    total_train_statistics = Statistics()
    report_train_statistics = Statistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []

    total_batch = -1
    early_stop_flag = False

    best_valid_ppl = float('inf')
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

    for epoch in range(opt.start_epoch, opt.epochs+1):
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
                #logging.info("one_batch")
                #report_loss.append(('train_ml_loss', loss_ml))
                #report_loss.append(('PPL', loss_ml))

                # Brief report
                '''
                if batch_i % opt.report_every == 0:
                    brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt)
                '''
            else:
                # TODO: traing_rl
                generator = SequenceGenerator(model,
                                              bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                              eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                              pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                              max_sequence_length=opt.max_length,
                                              copy_attn=opt.copy_attention,
                                              coverage_attn=opt.coverage_attn,
                                              length_penalty_factor=opt.length_penalty_factor,
                                              coverage_penalty_factor=opt.coverage_penalty_factor,
                                              length_penalty=opt.length_penalty,
                                              coverage_penalty=opt.coverage_penalty,
                                              cuda=opt.gpuid > -1
                                              )
                #train_rl.train_one_batch(batch, generator, optimizer_rl, opt)
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
            # Save the model parameters if the validation loss improved.
            if total_batch % 4000 == 0:
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            if opt.train_ml and epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
                    # test the model on the validation dataset for one epoch
                    valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    print("Enter check point!")
                    sys.stdout.flush()

                    current_train_ppl = report_train_statistics.ppl()
                    current_train_loss = report_train_statistics.xent()

                    # debug
                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                        exit()

                    if current_valid_loss < best_valid_loss: # update the best valid loss and save the model parameters
                        print("Valid loss drops")
                        sys.stdout.flush()
                        best_valid_loss = current_valid_loss
                        best_valid_ppl = current_valid_ppl
                        num_stop_dropping = 0

                        check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                            opt.exp, epoch, batch_i, total_batch) + '.model')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)

                    else:
                        print("Valid loss does not drop")
                        sys.stdout.flush()
                        num_stop_dropping += 1
                        # decay the learning rate by a factor
                        for i, param_group in enumerate(optimizer_ml.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr

                    # log loss, ppl, and time
                    #print("check point!")
                    #sys.stdout.flush()
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))
                    '''
                    train_forward_time, train_loss_compute_time, train_backward_time = report_train_statistics.total_time()
                    valid_forward_time, valid_loss_compute_time, _ = valid_loss_stat.total_time()
                    logging.info('avg. training forward time: %.1f; avg. training loss compute time: %.1f; avg. training backward time: %.1f' % (
                        train_forward_time, train_loss_compute_time, train_backward_time
                    ))
                    logging.info('avg. validation forward time: %.1f; avg. validation loss compute time: %.1f' % (
                        valid_forward_time, valid_loss_compute_time
                    ))
                    '''
                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_valid_ppl)
                    report_train_loss.append(current_train_loss)
                    report_valid_loss.append(current_valid_loss)

                    if num_stop_dropping >= opt.early_stop_tolerance:
                        logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)
                        early_stop_flag = True
                        break
                    report_train_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_results(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl, opt.checkpoint_interval, train_valid_curve_path)
    #logging.info('Overall average training loss: %.3f, ppl: %.3f' % (total_train_statistics.xent(), total_train_statistics.ppl()))

def main(opt):
    try:
        start_time = time.time()
        train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)
        start_time = time.time()
        model = init_model(opt)
        optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)
        train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt)
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

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)

