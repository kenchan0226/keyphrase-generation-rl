import torch.nn as nn
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.time_log import time_since
from evaluate import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
from utils.report import export_train_and_valid_loss
from utils.source_representation_queue import SourceRepresentationQueue
import numpy as np

EPS = 1e-8

def train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt):
    '''
    generator = SequenceGenerator(model,
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )
    '''
    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    num_stop_dropping = 0

    if opt.use_target_encoder:
        source_representation_queue = SourceRepresentationQueue(opt.source_representation_queue_size)
    else:
        source_representation_queue = None

    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    model.train()

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        # TODO: progress bar
        #progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,total_examples=len(train_data_loader.dataset.examples))

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            # Training
            if opt.train_ml:
                batch_loss_stat, decoder_dist = train_one_batch(batch, model, optimizer_ml, opt, batch_i, source_representation_queue)
                report_train_loss_statistics.update(batch_loss_stat)
                total_train_loss_statistics.update(batch_loss_stat)
                #logging.info("one_batch")
                #report_loss.append(('train_ml_loss', loss_ml))
                #report_loss.append(('PPL', loss_ml))

                # Brief report
                '''
                if batch_i % opt.report_every == 0:
                    brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt)
                '''

            #progbar.update(epoch, batch_i, report_loss)

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if total_batch % 4000 == 0:
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
                    if opt.train_ml:
                        # test the model on the validation dataset for one epoch
                        valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                        model.train()
                        current_valid_loss = valid_loss_stat.xent()
                        current_valid_ppl = valid_loss_stat.ppl()
                        print("Enter check point!")
                        sys.stdout.flush()

                        current_train_ppl = report_train_loss_statistics.ppl()
                        current_train_loss = report_train_loss_statistics.xent()

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

                        report_train_ppl.append(current_train_ppl)
                        report_valid_ppl.append(current_valid_ppl)
                        report_train_loss.append(current_train_loss)
                        report_valid_loss.append(current_valid_loss)

                        if num_stop_dropping >= opt.early_stop_tolerance:
                            logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)
                            early_stop_flag = True
                            break
                        report_train_loss_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl, opt.checkpoint_interval, train_valid_curve_path)
    #logging.info('Overall average training loss: %.3f, ppl: %.3f' % (total_train_loss_statistics.xent(), total_train_loss_statistics.ppl()))

def train_one_batch(batch, model, optimizer, opt, batch_i, source_representation_queue=None):
    if not opt.one2many:  # load one2one data
        src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, title, title_oov, title_lens, title_mask = batch
        """
        src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        src_lens: a list containing the length of src sequences for each batch, with len=batch
        src_mask: a FloatTensor, [batch, src_seq_len]
        trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        trg_lens: a list containing the length of trg sequences for each batch, with len=batch
        trg_mask: a FloatTensor, [batch, trg_seq_len]
        src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        trg_oov: a LongTensor containing the word indices of target sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        """
    else:  # load one2many data
        src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
        num_trgs = [len(trg_str_list) for trg_str_list in trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size
        """
        trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[pykp.io.SEP_WORD]
             if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eos>
        trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
        """
    batch_size = src.size(0)
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)
    if opt.title_guided:
        title = title.to(opt.device)
        title_mask = title_mask.to(opt.device)
        #title_oov = title_oov.to(opt.device)
    # title, title_oov, title_lens, title_mask

    optimizer.zero_grad()

    #if opt.one2many_mode == 0 or opt.one2many_mode == 1:
    start_time = time.time()

    if opt.use_target_encoder:  # Sample encoder representations
        if len(source_representation_queue) < opt.source_representation_sample_size:
            source_representation_samples_2dlist = None
            source_representation_target_list = None
        else:
            source_representation_samples_2dlist = []
            source_representation_target_list = []
            for i in range(batch_size):
                # N encoder representation from the queue
                source_representation_samples_list = source_representation_queue.sample(opt.source_representation_sample_size)
                # insert a place-holder for the ground-truth source representation to a random index
                place_holder_idx = np.random.randint(0, opt.source_representation_sample_size+1)
                source_representation_samples_list.insert(place_holder_idx, None)  # len=N+1
                # insert the sample list of one batch to the 2d list
                source_representation_samples_2dlist.append(source_representation_samples_list)
                # store the idx of place-holder for that batch
                source_representation_target_list.append(place_holder_idx)
    else:
        source_representation_samples_2dlist = None
        source_representation_target_list = None

        """
        if encoder_representation_samples_2dlist[0] is None and batch_i > math.ceil(
                opt.encoder_representation_sample_size / batch_size):
            # a return value of none indicates we don't have sufficient samples
            # it will only occurs in the first few training steps
            raise ValueError("encoder_representation_samples should not be none at this batch!")
        """

    if not opt.one2many:
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens, source_classification_dist = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, sampled_source_representation_2dlist=source_representation_samples_2dlist, source_representation_target_list=source_representation_target_list, title=title, title_lens=title_lens, title_mask=title_mask)
    else:
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens, source_classification_dist = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, num_trgs=num_trgs, sampled_source_representation_2dlist=source_representation_samples_2dlist, source_representation_target_list=source_representation_target_list, title=title, title_lens=title_lens, title_mask=title_mask)
    forward_time = time_since(start_time)

    if opt.use_target_encoder:  # Put all the encoder final states to the queue. Need to call detach() first
        # encoder_final_state: [batch, memory_bank_size]
        [source_representation_queue.put(encoder_final_state[i, :].detach()) for i in range(batch_size)]

    start_time = time.time()
    if opt.copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                         opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss, delimiter_decoder_states, opt.orthogonal_loss, opt.lambda_orthogonal, delimiter_decoder_states_lens)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss, delimiter_decoder_states, opt.orthogonal_loss, opt.lambda_orthogonal, delimiter_decoder_states_lens)

    loss_compute_time = time_since(start_time)

    #else:  # opt.one2many_mode == 2
    #    forward_time = 0
    #    loss_compute_time = 0
    #    # TODO: a for loop to accumulate loss for each keyphrase
    #    # TODO: meanwhile, accumulate the forward time and loss_compute time
    #    pass

    total_trg_tokens = sum(trg_lens)

    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print(src_oov)
        print(src_str_list)
        print(src_lens)
        print(src_mask)
        print("trg")
        print(trg)
        print(trg_oov)
        print(trg_str_2dlist)
        print(trg_lens)
        print(trg_mask)
        print("oov list")
        print(oov_lists)
        print("Decoder")
        print(decoder_dist)
        print(h_t)
        print(attention_dist)
        raise ValueError("Loss is NaN")

    if opt.loss_normalization == "tokens": # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')

    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    # back propagation on the normalized loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        # grad_norm_after_clipping = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        # logging.info('clip grad (%f -> %f)' % (grad_norm_before_clipping, grad_norm_after_clipping))

    optimizer.step()

    # Compute target encoder loss
    if opt.use_target_encoder and source_classification_dist is not None:
        start_time = time.time()
        optimizer.zero_grad()
        # convert source_representation_target_list to a LongTensor with size=[batch_size, max_num_delimiters]
        max_num_delimiters = delimiter_decoder_states.size(2)
        source_representation_target = torch.LongTensor(source_representation_target_list).to(trg.device)  # [batch_size]
        # expand along the second dimension, since for the target for each delimiter states in the same batch are the same
        source_representation_target = source_representation_target.view(-1, 1).repeat(1, max_num_delimiters)  # [batch_size, max_num_delimiters]
        # mask for source representation classification
        source_representation_target_mask = torch.zeros(batch_size, max_num_delimiters).to(trg.device)
        for i in range(batch_size):
            source_representation_target_mask[i, :delimiter_decoder_states_lens[i]].fill_(1)
        # compute the masked loss
        loss_te = masked_cross_entropy(source_classification_dist, source_representation_target, source_representation_target_mask)
        loss_compute_time += time_since(start_time)
        # back propagation on the normalized loss
        start_time = time.time()
        loss_te.div(normalization).backward()
        backward_time += time_since(start_time)

        if opt.max_grad_norm > 0:
            grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

        optimizer.step()

    # construct a statistic object for the loss
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat, decoder_dist.detach()
