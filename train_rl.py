import torch
import numpy as np
import pykp.io
import torch.nn as nn
from utils.statistics import RewardStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_train_and_valid_loss, export_train_and_valid_reward
import sys
import logging
import os
from evaluate import evaluate_reward
from pykp.reward import *
import math

EPS = 1e-8

def train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt):
    total_batch = -1
    early_stop_flag = False

    report_train_reward_statistics = RewardStatistics()
    total_train_reward_statistics = RewardStatistics()
    report_train_reward = []
    report_valid_reward = []
    best_valid_reward = float('-inf')
    num_stop_increasing = 0
    init_perturb_std = opt.init_perturb_std
    final_perturb_std = opt.final_perturb_std
    perturb_decay_factor = opt.perturb_decay_factor
    perturb_decay_mode = opt.perturb_decay_mode
    perturb_decay_along_phrases = opt.perturb_decay_along_phrases

    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  beam_size=1,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  cuda=opt.gpuid > -1
                                  )

    model.train()

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        # TODO: progress bar
        # progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,total_examples=len(train_data_loader.dataset.examples))
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(-1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1+total_batch)/4000))

            batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std, perturb_decay_along_phrases)
            report_train_reward_statistics.update(batch_reward_stat)
            total_train_reward_statistics.update(batch_reward_stat)

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if total_batch % 4000 == 0:
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):

                    valid_reward_stat = evaluate_reward(valid_data_loader, generator, opt)
                    model.train()
                    current_valid_reward = valid_reward_stat.reward()
                    print("Enter check point!")
                    sys.stdout.flush()

                    current_train_reward = report_train_reward_statistics.reward()
                    current_train_pg_loss = report_train_reward_statistics.loss()

                    if current_valid_reward > best_valid_reward:
                        print("Valid reward increases")
                        sys.stdout.flush()
                        best_valid_reward = current_valid_reward
                        num_stop_increasing = 0

                        check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                            opt.exp, epoch, batch_i, total_batch) + '.model')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)
                    else:
                        print("Valid reward does not increase")
                        sys.stdout.flush()
                        num_stop_increasing += 1
                        # decay the learning rate by the factor specified by opt.learning_rate_decay
                        if opt.learning_rate_decay_rl:
                            for i, param_group in enumerate(optimizer_rl.param_groups):
                                old_lr = float(param_group['lr'])
                                new_lr = old_lr * opt.learning_rate_decay
                                if old_lr - new_lr > EPS:
                                    param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training reward: %.4f; avg training loss: %.4f; avg validation reward: %.4f; best validation reward: %.4f' % (
                            current_train_reward, current_train_pg_loss, current_valid_reward, best_valid_reward))

                    report_train_reward.append(current_train_reward)
                    report_valid_reward.append(current_valid_reward)

                    if opt.early_stop_rl:
                        if num_stop_increasing >= opt.early_stop_tolerance:
                            logging.info('Have not increased for %d check points, early stop training' % num_stop_increasing)
                            early_stop_flag = True
                            break
                    report_train_reward_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_reward(report_train_reward, report_valid_reward, opt.checkpoint_interval, train_valid_curve_path)


def train_one_batch(one2many_batch, generator, optimizer, opt, perturb_std=0, perturb_decay_along_phrases=False):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = one2many_batch
    """
    src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    src_lens: a list containing the length of src sequences for each batch, with len=batch
    src_mask: a FloatTensor, [batch, src_seq_len]
    src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    oov_lists: a list of oov words for each src, 2dlist
    """

    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    # trg = trg.to(opt.device)
    # trg_mask = trg_mask.to(opt.device)
    # trg_oov = trg_oov.to(opt.device)

    optimizer.zero_grad()

    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type
    reward_shaping = opt.reward_shaping
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_factor = opt.regularization_factor

    if opt.perturb_baseline:
        baseline_perturb_std = perturb_std
    else:
        baseline_perturb_std = 0

    #generator.model.train()

    # sample a sequence from the model
    # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
    # log_selected_token_dist: size: [batch, output_seq_len]
    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, perturb_decay_along_phrases=perturb_decay_along_phrases)
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word)
    sample_time = time_since(start_time)
    max_pred_seq_len = log_selected_token_dist.size(1)

    # if use self critical as baseline, greedily decode a sequence from the model
    if opt.baseline == 'self':
        generator.model.eval()
        with torch.no_grad():
            start_time = time.time()
            greedy_sample_list, _, _, greedy_eos_idx_mask = generator.sample(src, src_lens, src_oov, src_mask,
                                                                             oov_lists, opt.max_length,
                                                                             greedy=True, one2many=one2many,
                                                                             one2many_mode=one2many_mode,
                                                                             num_predictions=num_predictions,
                                                                             perturb_std=baseline_perturb_std)
            greedy_str_2dlist = sample_list_to_str_2dlist(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size,
                                                          eos_idx,
                                                          delimiter_word)
        generator.model.train()
    '''
    if opt.pg_method == 0:
        # reward: an np array with size [batch_size]
        reward = compute_reward(trg_str_2dlist, pred_str_2dlist, batch_size, reward_type, topk)
        generator.model.eval()
        with torch.no_grad():
            greedy_sample_list, _, _, greedy_eos_idx_mask = generator.sample(src, src_lens, src_oov, src_mask,
                                                                             oov_lists, opt.max_length,
                                                                             greedy=True, one2many=one2many, one2many_mode=one2many_mode, num_predictions=num_predictions)
            greedy_str_2dlist = sample_list_to_str_2dlist(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                    delimiter_word)
        generator.model.train()
        baseline = compute_reward(trg_str_2dlist, greedy_str_2dlist, batch_size, reward_type, topk)
        baselined_reward = reward - baseline
        baselined_reward = np.repeat(baselined_reward[:, np.newaxis], max_pred_seq_len, axis=1)  # [batch_size, prediction_seq_len]
        assert baselined_reward.shape == (batch_size, max_pred_seq_len)
        q_value_sample = torch.from_numpy(baselined_reward).type(torch.FloatTensor).to(src.device)
        q_value_sample.requires_grad_(True)
        final_reward = reward

    elif opt.pg_method == 1:  # stepwise reward
        #reward = np.zeros((batch_size, max_pred_seq_len))
        phrase_reward = np.zeros((batch_size, num_predictions + 1))  # store the reward received for each prediction, the last column is the reward for padded words, which must be 0
        for t in range(num_predictions):
            pred_str_2dlist_at_t = [pred_str_list[:t+1] for pred_str_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_reward(trg_str_2dlist, pred_str_2dlist_at_t, batch_size, reward_type, topk)
        with torch.no_grad():
            greedy_sample_list, _, _, greedy_eos_idx_mask = generator.sample(src, src_lens, src_oov, src_mask,
                                                                              oov_lists, opt.max_length,
                                                                              greedy=True, one2many=one2many,
                                                                              one2many_mode=one2many_mode,
                                                                              num_predictions=num_predictions)
            greedy_str_2dlist = sample_list_to_str_2dlist(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size,
                                                          eos_idx,
                                                          delimiter_word)
        generator.model.train()
        phrase_baseline = np.zeros((batch_size, num_predictions + 1))
        for t in range(num_predictions):
            greedy_str_2dlist_at_t = [greedy_str_list[:t + 1] for greedy_str_list in greedy_str_2dlist]
            phrase_baseline[:, t] = compute_reward(trg_str_2dlist, greedy_str_2dlist_at_t, batch_size, reward_type, topk)
        baselined_phrase_reward = phrase_reward - phrase_baseline
        baselined_phrase_reward = torch.from_numpy(baselined_phrase_reward).type(torch.FloatTensor).to(src.device).requires_grad_(False)
        baselined_reward = torch.gather(baselined_phrase_reward, dim=1, index=pred_phrase_idx_mask)
        q_value_sample = baselined_reward

        q_value_sample.requires_grad_(True)
        final_reward = phrase_reward[:, num_predictions - 1]
    '''

    # Compute the reward for each predicted keyphrase
    # if using reward shaping, each keyphrase will have its own reward, else, only the last keyphrase will get a reward
    phrase_reward = compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, num_predictions, reward_shaping,
                          reward_type, topk, match_type, regularization_factor)  # np array with size: [batch_size, num_predictions]
    cumulative_reward = phrase_reward[:, num_predictions - 1]
    cumulative_reward_sum = cumulative_reward.sum(0)

    # Subtract reward by a baseline if needed
    if opt.baseline == 'self':
        phrase_baseline = compute_phrase_reward(greedy_str_2dlist, trg_str_2dlist, batch_size, num_predictions, reward_shaping,
                          reward_type, topk, match_type, regularization_factor)
        phrase_reward = phrase_reward - phrase_baseline

    if reward_shaping:
        phrase_reward = shape_reward(phrase_reward)

    # convert to reward received at each decoding step
    stepwise_reward = phrase_reward_to_stepwise_reward(phrase_reward, pred_eos_idx_mask)

    #shapped_baselined_reward = torch.gather(shapped_baselined_phrase_reward, dim=1, index=pred_phrase_idx_mask)

    # use the return as the estimation of q_value at each step
    q_value_estimate = np.cumsum(stepwise_reward[:,::-1], axis=1)[:,::-1].copy()
    q_value_estimate = torch.from_numpy(q_value_estimate).type(torch.FloatTensor).to(src.device)
    q_value_estimate.requires_grad_(True)
    q_estimate_compute_time = time_since(start_time)

    # compute the policy gradient objective
    pg_loss = compute_pg_loss(log_selected_token_dist, output_mask, q_value_estimate)

    # back propagation to compute the gradient
    start_time = time.time()
    pg_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(generator.model.parameters(), opt.max_grad_norm)

    # take a step of gradient descent
    optimizer.step()

    stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), batch_size, sample_time, q_estimate_compute_time, backward_time)
    # (final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0)
    # reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0

    return stat, log_selected_token_dist.detach()

'''
def preprocess_sample_list(sample_list, idx2word, vocab_size, oov_lists, eos_idx):
    for sample, oov in zip(sample_list, oov_lists):
        sample['sentence'] = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx)
    return
'''
