import numpy as np
from utils.string_helper import *
from evaluate_prediction import compute_match_result, compute_classification_metrics_at_k, check_duplicate_keyphrases, ndcg_at_k
import torch


def sample_list_to_str_2dlist(sample_list, oov_lists, idx2word, vocab_size, eos_idx, delimiter_word):
    """Convert a list of sample dict to a 2d list of predicted keyphrases"""
    pred_str_2dlist = []  #  a 2dlist, len(pred_str_2d_list)=batch_size, len(pred_str_2d_list[0])=
    for sample, oov in zip(sample_list, oov_lists):
        word_list = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx)
        pred_str_list = split_concated_keyphrases(word_list, delimiter_word)
        pred_str_2dlist.append(pred_str_list)
    return pred_str_2dlist


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


def compute_pg_loss(log_likelihood, output_mask, q_val_sample):
    """
    :param log_likelihood: [batch_size, prediction_seq_len]
    :param input_mask: [batch_size, prediction_seq_len]
    :param q_val_sample: [batch_size, prediction_seq_len]
    :return:
    """
    log_likelihood = log_likelihood.view(-1)  # [batch_size * prediction_seq_len]
    output_mask = output_mask.view(-1)  # [batch_size * prediction_seq_len]
    q_val_sample = q_val_sample.view(-1)  # [batch_size * prediction_seq_len]
    objective = -log_likelihood * output_mask * q_val_sample
    objective = torch.sum(objective)/torch.sum(output_mask)
    return objective
