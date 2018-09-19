import torch
from utils.string_helper import *
from evaluate_prediction import compute_match_result, compute_classification_metrics_at_k, check_duplicate_keyphrases, ndcg_at_k
from evaluate import split_concated_keyphrases
import numpy as np
import pykp.io
import torch.nn as nn

def train_one_batch(one2many_batch, generator, optimizer, opt):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = one2many_batch
    """
    src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    src_lens: a list containing the length of src sequences for each batch, with len=batch
    src_mask: a FloatTensor, [batch, src_seq_len]
    src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    oov_lists: a list of oov words for each src, 2dlist
    """
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type

    # sample a sequence
    # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
    # log_selected_token_dist: size: [batch, output_seq_len]
    sample_list, log_selected_token_dist, output_mask = generator.sample(src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False)
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word)
    max_pred_seq_len = log_selected_token_dist.size(1)

    if opt.baseline == 'self':
        # reward: an np array with size [batch_size]
        reward = compute_reward(trg_str_2dlist, pred_str_2dlist, batch_size, topk)
        greedy_sample_list, _, _ = generator.sample(src, src_lens, src_oov, src_mask,
                                                                             oov_lists, opt.max_length,
                                                                             greedy=True)
        greedy_str_2dlist = sample_list_to_str_2dlist(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                    delimiter_word)
        baseline_reward = compute_reward(trg_str_2dlist, greedy_str_2dlist, batch_size, reward_type, topk)
        reward = reward - baseline_reward
        reward = np.repeat(reward[:, np.newaxis], max_pred_seq_len)  # [batch_size, prediction_seq_len]
        assert reward.shape == (batch_size, max_pred_seq_len)
        q_value_sample = torch.from_numpy(reward).to(src.device)
        q_value_sample.requires_grad = True

    pg_loss = compute_pg_loss(log_selected_token_dist, output_mask, q_value_sample)
    pg_loss.backward()
    grad_norm_before_clipping = nn.utils.clip_grad_norm_(generator.model.parameters(), opt.max_grad_norm)
    optimizer.step()

    return

def sample_list_to_str_2dlist(sample_list, oov_lists, idx2word, vocab_size, eos_idx, delimiter_word):
    """Convert a list of sample dict to a 2d list of predicted keyphrases"""
    pred_str_2dlist = []  #  a 2dlist, len(pred_str_2d_list)=batch_size, len(pred_str_2d_list[0])=
    for sample, oov in zip(sample_list, oov_lists):
        word_list = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx)
        pred_str_list = split_concated_keyphrases(word_list, delimiter_word)
        pred_str_2dlist.append(pred_str_list)
    return pred_str_2dlist

def compute_self_critical_reward(sample_str_2dlist, greedy_str_2dlist):

    return

def compute_reward(trg_str_2dlist, pred_str_2dlist, batch_size, reward_type='f1', topk=10):
    assert len(trg_str_2dlist) == batch_size
    assert len(pred_str_2dlist) == batch_size
    reward = np.zeros(batch_size)
    for idx, (trg_str_list, pred_str_list) in enumerate(zip(trg_str_2dlist, pred_str_2dlist)):
        # perform stemming
        stemmed_trg_str_list = stem_str_list(trg_str_list)
        stemmed_pred_str_list = stem_str_list(pred_str_list)

        trg_str_filter = check_duplicate_keyphrases(stemmed_trg_str_list)  # a boolean nparray, true if not duplicated
        pred_str_filter = check_duplicate_keyphrases(stemmed_pred_str_list)

        unique_stemmed_trg_str_list = [word_list for word_list, is_keep in zip(stemmed_trg_str_list, trg_str_filter) if
                                 is_keep]
        unique_stemmed_pred_str_list = [word_list for word_list, is_keep in zip(stemmed_pred_str_list, pred_str_filter) if
                                  is_keep]
        num_unique_targets = len(unique_stemmed_trg_str_list)
        num_unique_predictions = len(unique_stemmed_pred_str_list)
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list, pred_str_list=unique_stemmed_pred_str_list)
        if reward_type == "f1":
            precision_k, recall_k, f1_k, _, _ = compute_classification_metrics_at_k(is_match, num_unique_predictions, num_unique_targets, topk=topk)
            reward[idx] = f1_k
        else:
            ndcg_k = ndcg_at_k(is_match, topk, method=1)
            reward[idx] = ndcg_k
    return reward

'''
def preprocess_sample_list(sample_list, idx2word, vocab_size, oov_lists, eos_idx):
    for sample, oov in zip(sample_list, oov_lists):
        sample['sentence'] = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx)
    return
'''

def compute_pg_loss(log_likelihood, output_mask, q_val_sample):
    """
    :param log_likelihood: [batch_size, prediction_seq_len]
    :param input_mask: [batch_size, prediction_seq_len]
    :param q_val_sample: [batch_size, prediction_seq_len]
    :return:
    """
    log_likelihood = log_likelihood.view(-1)  # [batch_size * prediction_seq_len]
    output_mask = output_mask.view(-1)  # [batch_size * prediction_seq_len]
    q_val_sample  # [batch_size * prediction_seq_len]
    objective = -log_likelihood * output_mask * q_val_sample
    objective = torch.sum(objective)/torch.sum(output_mask)
    return objective
