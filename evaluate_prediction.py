import numpy as np
import argparse
import config
from utils.string_helper import *
from collections import defaultdict
import os
import logging
import pykp.io


def check_valid_keyphrases(str_list):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if w == pykp.io.UNK_WORD or w == ',' or w == '.':
                keep_flag = False

        is_valid[i] = keep_flag

    return is_valid


def dummy_filter(str_list):
    num_pred_seq = len(str_list)
    return np.ones(num_pred_seq, dtype=bool)


def compute_extra_one_word_seqs_mask(str_list):
    num_pred_seq = len(str_list)
    mask = np.zeros(num_pred_seq, dtype=bool)
    num_one_word_seqs = 0
    for i, word_list in enumerate(str_list):
        if len(word_list) == 1:
            num_one_word_seqs += 1
            if num_one_word_seqs > 1:
                mask[i] = False
                continue
        mask[i] = True
    return mask, num_one_word_seqs


def check_duplicate_keyphrases(keyphrase_str_list):
    num_keyphrases = len(keyphrase_str_list)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()
    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        if '_'.join(keyphrase_word_list) in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True
        keyphrase_set.add('_'.join(keyphrase_word_list))
    return not_duplicate


def check_present_and_duplicate_keyphrases(src_str, keyphrase_str_list):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """

    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        if '_'.join(keyphrase_word_list) in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True

        # check if it appears in source text
        for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
            match = True
            for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                src_w = src_str[src_start_idx + keyphrase_i]
                if src_w != keyphrase_w:
                    match = False
                    break
            if match:
                break

        if match:
            is_present[i] = True
        else:
            is_present[i] = False
        keyphrase_set.add('_'.join(keyphrase_word_list))

    return is_present, not_duplicate


def compute_match_result_backup(trg_str_list, pred_str_list, type='exact'):
    assert type in ['exact', 'sub'], "Right now only support exact matching and substring matching"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    is_match = np.zeros(num_pred_str, dtype=bool)

    for pred_idx, pred_word_list in enumerate(pred_str_list):
        if type == 'exact':  # exact matching
            is_match[pred_idx] = False
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                if len(pred_word_list) != len(trg_word_list): # if length not equal, it cannot be a match
                    continue
                match = True
                for pred_w, trg_w in zip(pred_word_list, trg_word_list):
                    if pred_w != trg_w:
                        match = False
                        break
                # If there is one exact match in the target, match succeeds, go the next prediction
                if match:
                    is_match[pred_idx] = True
                    break
        elif type == 'sub':  # consider a match if the prediction is a subset of the target
            joined_pred_word_list = ' '.join(pred_word_list)
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                if joined_pred_word_list in ' '.join(trg_word_list):
                    is_match[pred_idx] = True
                    break
    return is_match


def compute_match_result(trg_str_list, pred_str_list, type='exact', dimension=1):
    assert type in ['exact', 'sub'], "Right now only support exact matching and substring matching"
    assert dimension in [1, 2], "only support 1 or 2"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    if dimension == 1:
        is_match = np.zeros(num_pred_str, dtype=bool)
        for pred_idx, pred_word_list in enumerate(pred_str_list):
            joined_pred_word_list = ' '.join(pred_word_list)
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                joined_trg_word_list = ' '.join(trg_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
    else:
        is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            for pred_idx, pred_word_list in enumerate(pred_str_list):
                joined_pred_word_list = ' '.join(pred_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
    return is_match


def prepare_classification_result_dict(precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, num_targets_k, topk, is_present):
    present_tag = "present" if is_present else "absent"
    return {'precision@%d_%s' % (topk, present_tag): precision_k, 'recall@%d_%s' % (topk, present_tag): recall_k,
            'f1_score@%d_%s' % (topk, present_tag): f1_k, 'num_matches@%d_%s' % (topk, present_tag): num_matches_k,
            'num_predictions@%d_%s' % (topk, present_tag): num_predictions_k, 'num_targets@%d_%s' % (topk, present_tag): num_targets_k}


def compute_classification_metrics_at_k(is_match, num_predictions, num_trgs, topk=5):
    """
    :param is_match: a boolean np array with size [num_predictions]
    :param predicted_list:
    :param true_list:
    :param topk:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions

    if num_predictions > topk:
        is_match = is_match[:topk]
        num_predictions_k = topk

    num_matches_k = sum(is_match)

    precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_k, num_predictions_k, num_trgs)

    return precision_k, recall_k, f1_k, num_matches_k, num_predictions_k


def compute_classification_metrics_at_ks(is_match, num_predictions, num_trgs, k_list=[5,10]):
    """
    :param is_match: a boolean np array with size [num_predictions]
    :param predicted_list:
    :param true_list:
    :param topk:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    #topk.sort()
    if num_predictions == 0:
        precision_ks = [0] * len(k_list)
        recall_ks = [0] * len(k_list)
        f1_ks = [0] * len(k_list)
        num_matches_ks = [0] * len(k_list)
        num_predictions_ks = [0] * len(k_list)
    else:
        num_matches = np.cumsum(is_match)
        num_predictions_ks = []
        num_matches_ks = []
        precision_ks = []
        recall_ks = []
        f1_ks = []
        for topk in k_list:
            if num_predictions > topk:
                num_matches_at_k = num_matches[topk-1]
                num_predictions_at_k = topk
            else:
                num_matches_at_k = num_matches[-1]
                num_predictions_at_k = num_predictions

            precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_at_k, num_predictions_at_k, num_trgs)
            precision_ks.append(precision_k)
            recall_ks.append(recall_k)
            f1_ks.append(f1_k)
            num_matches_ks.append(num_matches_at_k)
            num_predictions_ks.append(num_predictions_at_k)
    return precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks


def compute_classification_metrics(num_matches, num_predictions, num_trgs):
    precision = compute_precision(num_matches, num_predictions)
    recall = compute_recall(num_matches, num_trgs)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1


def compute_precision(num_matches, num_predictions):
    return num_matches / num_predictions if num_predictions > 0 else 0.0


def compute_recall(num_matches, num_trgs):
    return num_matches / num_trgs if num_trgs > 0 else 0.0


def compute_f1(precision, recall):
    return float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0


def dcg_at_k(r, k, method=1):
    """
    Reference from https://www.kaggle.com/wendykan/ndcg-example and https://gist.github.com/bwhite/3726239
    Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    num_predictions = r.shape[0]
    if num_predictions == 0:
        dcg = 0.
    else:
        if num_predictions > k:
            r = r[:k]
            num_predictions = k
        if method == 0:
            dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.sum(discounted_gain)
        else:
            raise ValueError('method must be 0 or 1.')
    return dcg


def dcg_at_ks(r, k_list, method=1):
    num_predictions = r.shape[0]
    if num_predictions == 0:
        dcg_array = np.array([0] * len(k_list))
    else:
        k_max = max(k_list)
        if num_predictions > k_max:
            r = r[:k_max]
            num_predictions = k_max
        if method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.cumsum(discounted_gain)
            return_indices = []
            for k in k_list:
                return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
            return_indices = np.array(return_indices, dtype=int)
            dcg_array = dcg[return_indices]
        else:
            raise ValueError('method must 1.')
    return dcg_array


def ndcg_at_k(r, k, method=1, include_dcg=False):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    if r.shape[0] == 0:
        ndcg = 0.0
        dcg = 0.0
    else:
        dcg_max = dcg_at_k(np.array(sorted(r, reverse=True)), k, method)
        dcg = dcg_at_k(r, k, method)
        ndcg = dcg / dcg_max
    if include_dcg:
        return ndcg, dcg
    else:
        return ndcg


def ndcg_at_ks(r, k_list, method=1, include_dcg=False):
    if r.shape[0] == 0:
        ndcg_array = [0.0] * len(k_list)
        dcg_array = [0.0] * len(k_list)
    else:
        dcg_array = dcg_at_ks(r, k_list, method)
        ideal_r = np.array(sorted(r, reverse=True))
        dcg_max_array = dcg_at_ks(ideal_r, k_list, method)
        ndcg_array = dcg_array / dcg_max_array
        ndcg_array = np.nan_to_num(ndcg_array)
    if include_dcg:
        return ndcg_array, dcg_array
    else:
        return ndcg_array


def alpha_dcg_at_k(r_2d, k, method=1, alpha=0.5):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_dcg = 0
    else:
        # convert r_2d to gain vector
        num_trg_str, num_pred_str = r_2d.shape
        if num_pred_str > k:
            num_pred_str = k
        gain_vector = np.zeros(num_pred_str)
        one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
        cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
        for j in range(num_pred_str):
            gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
        alpha_dcg = dcg_at_k(gain_vector, k, method)
    return alpha_dcg


def alpha_dcg_at_ks(r_2d, k_list, method=1, alpha=0.5):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param ks:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        return [0] * len(k_list)
    # convert r_2d to gain vector
    num_trg_str, num_pred_str = r_2d.shape
    k_max = max(k_list)
    if num_pred_str > k_max:
        num_pred_str = k_max
    gain_vector = np.zeros(num_pred_str)
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
    for j in range(num_pred_str):
        gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
    return dcg_at_ks(gain_vector, k_list, method)


def alpha_ndcg_at_k(r_2d, k, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg = 0
        alpha_dcg = 0
    else:
        # convert r to gain vector
        alpha_dcg = alpha_dcg_at_k(r_2d, k, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k, alpha)
        alpha_dcg_max = alpha_dcg_at_k(r_2d_ideal, k, method, alpha)
        alpha_ndcg = alpha_dcg / alpha_dcg_max
        alpha_ndcg = np.nan_to_num(alpha_ndcg)
    if include_dcg:
        return alpha_ndcg, alpha_dcg
    else:
        return alpha_ndcg


def alpha_ndcg_at_ks(r_2d, k_list, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg_array = [0] * len(k_list)
        alpha_dcg_array = [0] * len(k_list)
    else:
        k_max = max(k_list)
        # convert r to gain vector
        alpha_dcg_array = alpha_dcg_at_ks(r_2d, k_list, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k_max, alpha)
        alpha_dcg_max_array = alpha_dcg_at_ks(r_2d_ideal, k_list, method, alpha)
        alpha_ndcg_array = alpha_dcg_array / alpha_dcg_max_array
        alpha_ndcg_array = np.nan_to_num(alpha_ndcg_array)
    if include_dcg:
        return alpha_ndcg_array, alpha_dcg_array
    else:
        return alpha_ndcg_array


def compute_ideal_r_2d(r_2d, k, alpha=0.5):
    num_trg_str, num_pred_str = r_2d.shape
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r_vector = np.zeros((num_trg_str))
    ideal_ranking = []
    greedy_depth = min(num_pred_str, k)
    for rank in range(greedy_depth):
        gain_vector = np.zeros(num_pred_str)
        for j in range(num_pred_str):
            if j in ideal_ranking:
                gain_vector[j] = -1000.0
            else:
                gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r_vector))
        max_idx = np.argmax(gain_vector)
        ideal_ranking.append(max_idx)
        current_relevance_vector = r_2d[:, max_idx]
        cum_r_vector = cum_r_vector + current_relevance_vector
    return r_2d[:, np.array(ideal_ranking, dtype=int)]


def average_precision(r, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return 0
    r_cum_sum = np.cumsum(r, axis=0)
    precision_sum = sum([compute_precision(r_cum_sum[k], k + 1) for k in range(num_predictions) if r[k]])
    '''
    precision_sum = 0
    for k in range(num_predictions):
        if r[k] is False:
            continue
        else:
            precision_k = precision(r_cum_sum[k], k+1)
            precision_sum += precision_k
    '''
    return precision_sum / num_trgs


def average_precision_at_k(r, k, num_predictions, num_trgs):
    if k < num_predictions:
        num_predictions = k
        r = r[:k]
    return average_precision(r, num_predictions, num_trgs)


def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    k_max = max(k_list)
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        return_indices.append( (k-1) if k <= num_predictions else (num_predictions-1) )
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]


def main(opt):
    src_file_path = opt.src_file_path
    trg_file_path = opt.trg_file_path
    pred_file_path = opt.pred_file_path

    if opt.export_filtered_pred:
        pred_output_file = open(os.path.join(opt.filtered_pred_path, "predictions_filtered.txt"), "w")

    # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    score_dict_all = defaultdict(list)
    topk_dict = {'present': [5, 10], 'absent': [10, 50], 'all': [10, 20]}
    num_src = 0
    num_unique_predictions = 0
    num_unique_present_predictions = 0
    num_unique_absent_predictions = 0
    num_unique_present_filtered_predictions = 0
    num_unique_present_filtered_targets = 0
    num_unique_absent_filtered_predictions = 0
    num_unique_absent_filtered_targets = 0
    max_unique_targets = 0

    for data_idx, (src_l, trg_l, pred_l) in enumerate(zip(open(src_file_path), open(trg_file_path), open(pred_file_path))):
        num_src += 1
        pred_str_list = pred_l.strip().split(';')
        pred_str_list = pred_str_list[:opt.num_preds]
        pred_str_list = [pred_str.split(' ') for pred_str in pred_str_list]
        trg_str_list = trg_l.strip().split(';')
        trg_str_list = [trg_str.split(' ') for trg_str in trg_str_list]
        [title, context] = src_l.strip().split('<eos>')
        src_str = title.strip().split(' ') + context.strip().split(' ')
        stemmed_src_str = stem_word_list(src_str)
        stemmed_trg_str_list = stem_str_list(trg_str_list)
        stemmed_pred_str_list = stem_str_list(pred_str_list)

        # is_present: boolean np array indicate whether a predicted keyphrase is present in src
        # not_duplicate: boolean np array indicate
        trg_str_is_present, trg_str_not_duplicate = check_present_and_duplicate_keyphrases(stemmed_src_str,
                                                                                           stemmed_trg_str_list)
        current_unique_targets = sum(trg_str_not_duplicate)
        if current_unique_targets > max_unique_targets:
            max_unique_targets = current_unique_targets

        # a pred_seq is invalid if len(processed_seq) == 0 or keep_flag and any word in processed_seq is UNK or it contains '.' or ','
        if not opt.disable_valid_filter:
            pred_str_is_valid = check_valid_keyphrases(pred_str_list)
        else:
            pred_str_is_valid = dummy_filter(pred_str_list)
        # pred_str_is_valid, processed_pred_seq_list, processed_pred_str_list, processed_pred_score_list = process_predseqs(pred_seq_list, oov, opt.idx2word, opt)

        # a list of boolean indicates which predicted keyphrases present in src, for the duplicated keyphrases after stemming, only consider the first one
        pred_str_is_present, pred_str_not_duplicate = check_present_and_duplicate_keyphrases(stemmed_src_str,
                                                                                             stemmed_pred_str_list)
        num_unique_predictions += sum(pred_str_not_duplicate)

        # Only keep the first one-word prediction, remove all the remaining keyphrases that only has one word.
        if not opt.disable_extra_one_word_filter:
            extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_str_list)
        else:
            extra_one_word_seqs_mask = dummy_filter(pred_str_list)

        tmp_trg_str_filter = trg_str_not_duplicate
        tmp_pred_str_filter = pred_str_not_duplicate * pred_str_is_valid * extra_one_word_seqs_mask

        # Compute NDCG for the all results include both present and absent keyphrases
        filtered_stemmed_trg_str_list_all = [word_list for word_list, is_keep in
                                         zip(stemmed_trg_str_list, tmp_trg_str_filter)
                                         if
                                         is_keep]
        filtered_stemmed_pred_str_list_all = [word_list for word_list, is_keep in
                                          zip(stemmed_pred_str_list, tmp_pred_str_filter) if
                                          is_keep]
        num_filtered_targets_all = len(filtered_stemmed_trg_str_list_all)
        num_filtered_predictions_all = len(filtered_stemmed_pred_str_list_all)

        is_match_all = compute_match_result(filtered_stemmed_trg_str_list_all, filtered_stemmed_pred_str_list_all, type='exact', dimension=1)
        is_match_substring_2d_all = compute_match_result(filtered_stemmed_trg_str_list_all, filtered_stemmed_pred_str_list_all, type='sub', dimension=2)

        precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
            compute_classification_metrics_at_ks(is_match_all, num_filtered_predictions_all, num_filtered_targets_all, k_list=topk_dict['all'])

        ndcg_ks, dcg_ks = ndcg_at_ks(is_match_all, k_list=topk_dict['all'], method=1, include_dcg=True)
        alpha_ndcg_ks, alpha_dcg_ks = alpha_ndcg_at_ks(is_match_substring_2d_all, k_list=topk_dict['all'], method=1, alpha=0.5, include_dcg=True)

        ap_ks = average_precision_at_ks(is_match_all, k_list=topk_dict['all'], num_predictions=num_filtered_predictions_all, num_trgs=num_filtered_targets_all)

        for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ndcg_k, dcg_k, alpha_ndcg_k, alpha_dcg_k, ap_k in \
                zip(topk_dict['all'], precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ndcg_ks, dcg_ks, alpha_ndcg_ks, alpha_dcg_ks, ap_ks):
            score_dict_all['precision@%d_all' % (topk)].append(precision_k)
            score_dict_all['recall@%d_all' % (topk)].append(recall_k)
            score_dict_all['f1_score@%d_all' % (topk)].append(f1_k)
            score_dict_all['num_matches@%d_all' % (topk)].append(num_matches_k)
            score_dict_all['num_predictions@%d_all' % (topk)].append(num_predictions_k)
            score_dict_all['num_targets@%d_all' % (topk)].append(num_filtered_targets_all)
            score_dict_all['DCG@%d_all' % (topk)].append(dcg_k)
            score_dict_all['NDCG@%d_all' % (topk)].append(ndcg_k)
            score_dict_all['AlphaDCG@%d_all' % (topk)].append(alpha_dcg_k)
            score_dict_all['AlphaNDCG@%d_all' % (topk)].append(alpha_ndcg_k)
            score_dict_all['AP@%d_all' % (topk)].append(ap_k)

        # Filter for present keyphrase prediction
        trg_str_filter_present = tmp_trg_str_filter * trg_str_is_present
        pred_str_filter_present = tmp_pred_str_filter * pred_str_is_present

        # Filter for absent keyphrase prediction
        trg_str_filter_absent = tmp_trg_str_filter * np.invert(trg_str_is_present)
        pred_str_filter_absent = tmp_pred_str_filter * np.invert(pred_str_is_present)

        # Increment num of unique predictions for present and absent keyphrases
        num_unique_present_predictions += sum(pred_str_not_duplicate * pred_str_is_present)
        num_unique_absent_predictions += sum(pred_str_not_duplicate * np.invert(pred_str_is_present))

        # A list to store all the predicted keyphrases after filtering for both present and absent keyphrases
        filtered_pred_dict = {"present": None, "absent": None}

        for is_present in [True, False]:
            if is_present:
                present_tag = "present"
                trg_str_filter = trg_str_filter_present
                pred_str_filter = pred_str_filter_present
            else:
                present_tag = "absent"
                trg_str_filter = trg_str_filter_absent
                pred_str_filter = pred_str_filter_absent

            # Apply filter to
            filtered_trg_str_list = [word_list for word_list, is_keep in zip(trg_str_list, trg_str_filter) if
                                     is_keep]
            filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in
                                             zip(stemmed_trg_str_list, trg_str_filter)
                                             if
                                             is_keep]

            filtered_pred_str_list = [word_list for word_list, is_keep in zip(pred_str_list, pred_str_filter) if
                                      is_keep]
            filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                              zip(stemmed_pred_str_list, pred_str_filter) if
                                              is_keep]
            #filtered_pred_score_list = [score for score, is_keep in zip(pred_score_list, pred_str_filter) if is_keep]
            #filtered_pred_attn_list = [attn for attn, is_keep in zip(pred_attn_list, pred_str_filter) if is_keep]

            filtered_pred_dict[present_tag] = filtered_pred_str_list

            # A boolean np array indicates whether each prediction match the target after stemming
            is_match = compute_match_result(trg_str_list=filtered_stemmed_trg_str_list, pred_str_list=filtered_stemmed_pred_str_list)
            #is_match_2d = compute_match_result(filtered_stemmed_trg_str_list, filtered_stemmed_pred_str_list, type='exact', dimension=2)
            #is_match = np.sum(is_match_2d, axis=0)
            is_match_substring_2d = compute_match_result(trg_str_list=filtered_stemmed_trg_str_list, pred_str_list=filtered_stemmed_pred_str_list, type='sub', dimension=2)

            num_filtered_predictions = len(filtered_pred_str_list)
            num_filtered_targets = len(filtered_trg_str_list)
            if is_present:
                num_unique_present_filtered_predictions += num_filtered_predictions
                num_unique_present_filtered_targets += num_filtered_targets
            else:
                num_unique_absent_filtered_predictions += num_filtered_predictions
                num_unique_absent_filtered_targets += num_filtered_targets

            precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
                compute_classification_metrics_at_ks(is_match, num_filtered_predictions,
                                                    num_filtered_targets, k_list=topk_dict[present_tag])

            ndcg_ks, dcg_ks = ndcg_at_ks(is_match, k_list=topk_dict[present_tag], method=1, include_dcg=True)
            alpha_ndcg_ks, alpha_dcg_ks = alpha_ndcg_at_ks(is_match_substring_2d, k_list=topk_dict[present_tag], method=1,
                                                           alpha=0.5, include_dcg=True)
            ap_ks = average_precision_at_ks(is_match, k_list=topk_dict[present_tag],
                                            num_predictions=num_filtered_predictions,
                                            num_trgs=num_filtered_targets)

            for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ndcg_k, dcg_k, alpha_ndcg_k, alpha_dcg_k, ap_k in \
                    zip(topk_dict[present_tag], precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ndcg_ks, dcg_ks, alpha_ndcg_ks, alpha_dcg_ks, ap_ks):
                score_dict_all['precision@%d_%s' % (topk, present_tag)].append(precision_k)
                score_dict_all['recall@%d_%s' % (topk, present_tag)].append(recall_k)
                score_dict_all['f1_score@%d_%s' % (topk, present_tag)].append(f1_k)
                score_dict_all['num_matches@%d_%s' % (topk, present_tag)].append(num_matches_k)
                score_dict_all['num_predictions@%d_%s' % (topk, present_tag)].append(num_predictions_k)
                score_dict_all['num_targets@%d_%s' % (topk, present_tag)].append(num_filtered_targets)

                score_dict_all['DCG@%d_%s' % (topk, present_tag)].append(dcg_k)
                score_dict_all['NDCG@%d_%s' % (topk, present_tag)].append(ndcg_k)
                score_dict_all['AlphaDCG@%d_%s' % (topk, present_tag)].append(alpha_dcg_k)
                score_dict_all['AlphaNDCG@%d_%s' % (topk, present_tag)].append(alpha_ndcg_k)
                score_dict_all['AP@%d_%s' % (topk, present_tag)].append(ap_k)

        marco_avg_precison_10_present = sum(score_dict_all['precision@10_present']) / len(score_dict_all['precision@10_present'])
        micro_avg_precision_10_present = sum(score_dict_all['num_matches@10_present']) / sum(score_dict_all['num_predictions@10_present'])

        #print("Result dict all:")
        #print(score_dict_all)

        if opt.export_filtered_pred:
            pred_print_out = ''
            # print out present keyphrases, each of them separated by ';', then print ||
            for word_list_i, word_list in enumerate(filtered_pred_dict['present']):
                if word_list_i < len(filtered_pred_dict['present']) - 1:
                    pred_print_out += '%s;' % ' '.join(word_list)
                else:
                    pred_print_out += '%s|' % ' '.join(word_list)
            # print out absent keyphrases, each of them separated by ';',
            for word_list_i, word_list in enumerate(filtered_pred_dict['absent']):
                if word_list_i < len(filtered_pred_dict['absent']) - 1:
                    pred_print_out += '%s;' % ' '.join(word_list)
                else:
                    pred_print_out += '%s' % ' '.join(word_list)
            pred_print_out += '\n'
            pred_output_file.write(pred_print_out)


    if opt.export_filtered_pred:
        pred_output_file.close()

    logging.info('Total number of samples: %d' % num_src)
    logging.info('Total number of unique predictions: %d' % num_unique_predictions)

    num_unique_filtered_predictions = num_unique_present_filtered_predictions+num_unique_absent_filtered_predictions
    num_unique_filtered_targets = num_unique_present_filtered_targets+num_unique_absent_filtered_targets

    logging.info('Avg. filtered targets per src: %.2f' % (num_unique_filtered_targets/num_src))
    logging.info('Max. unique targets per src: %d' % (max_unique_targets))

    logging.info("====================All======================")
    for topk in topk_dict['all']:
        total_predictions = sum(score_dict_all['num_predictions@%d_all' % (topk)])
        total_targets = sum(score_dict_all['num_targets@%d_all' % (topk)])
        # Compute the micro averaged recall, precision and F-1 score
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(
            sum(score_dict_all['num_matches@%d_all' % (topk)]), total_predictions, total_targets)
        logging.info('micro_avg_precision@%d_all:%.5f' % (topk, micro_avg_precision_k))
        logging.info('micro_avg_recall@%d_all:%.5f' % (topk, micro_avg_recall_k))
        logging.info('micro_avg_f1_score@%d_all:%.5f' % (topk, micro_avg_f1_score_k))

        avg_dcg = sum(score_dict_all['DCG@%d_all' % (topk)]) / len(
            score_dict_all['DCG@%d_all' % (topk)])
        avg_ndcg = sum(score_dict_all['NDCG@%d_all' % (topk)]) / len(
            score_dict_all['NDCG@%d_all' % (topk)])
        avg_alpha_dcg = sum(score_dict_all['AlphaDCG@%d_all' % (topk)]) / len(
            score_dict_all['AlphaDCG@%d_all' % (topk)])
        avg_alpha_ndcg = sum(score_dict_all['AlphaNDCG@%d_all' % (topk)]) / len(
            score_dict_all['AlphaNDCG@%d_all' % (topk)])
        map = sum(score_dict_all['AP@%d_all' % (topk)]) / len(
            score_dict_all['AP@%d_all' % (topk)])
        logging.info('avg_DCG@%d_all: %.5f' % (topk, avg_dcg))
        logging.info('avg_NDCG@%d_all: %.5f' % (topk, avg_ndcg))
        logging.info('avg_Alpha_DCG@%d_all: %.5f' % (topk, avg_alpha_dcg))
        logging.info('avg_Alpha_NDCG@%d_all: %.5f' % (topk, avg_alpha_ndcg))
        logging.info('MAP@%d_all: %.5f' % (topk, map))

    for is_present in [True, False]:
        if is_present:
            present_tag = 'present'
            logging.info("====================Present======================")
            logging.info("Total number of unique present predictions: %d/%d" % (num_unique_present_predictions, num_unique_predictions))
            logging.info("Total number of unique present predictions after filtering: %d/%d" % (num_unique_present_filtered_predictions, num_unique_filtered_predictions))
            logging.info("Total number of present targets after filtering: %d/%d" % (num_unique_present_filtered_targets, num_unique_filtered_targets))
        else:
            present_tag = 'absent'
            logging.info("====================Absent======================")
            logging.info("Total number of unique absent predictions: %d/%d" % (num_unique_absent_predictions,num_unique_predictions))
            logging.info(
                "Total number of unique absent predictions after filtering: %d/%d" % (num_unique_absent_filtered_predictions, num_unique_filtered_predictions))
            logging.info("Total number of absent targets after filtering: %d/%d" % (num_unique_absent_filtered_targets, num_unique_filtered_targets))

        logging.info('Final Results (%s):' % present_tag)
        for topk in topk_dict[present_tag]:
            total_predictions = sum(score_dict_all['num_predictions@%d_%s' % (topk, present_tag)])
            total_targets = sum(score_dict_all['num_targets@%d_%s' % (topk, present_tag)])
            # Compute the micro averaged recall, precision and F-1 score
            micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(sum(score_dict_all['num_matches@%d_%s' % (topk, present_tag)]), total_predictions, total_targets)
            logging.info('micro_avg_precision@%d_%s:%.5f' % (topk, present_tag, micro_avg_precision_k))
            logging.info('micro_avg_recall@%d_%s:%.5f' % (topk, present_tag, micro_avg_recall_k))
            logging.info('micro_avg_f1_score@%d_%s:%.5f' % (topk, present_tag, micro_avg_f1_score_k))
            # Compute the macro averaged recall, precision and F-1 score
            macro_avg_precision_k = sum(score_dict_all['precision@%d_%s' % (topk, present_tag)])/len(score_dict_all['precision@%d_%s' % (topk, present_tag) ])
            marco_avg_recall_k = sum(score_dict_all['recall@%d_%s' % (topk, present_tag)])/len(score_dict_all['recall@%d_%s' % (topk, present_tag) ])
            marco_avg_f1_score_k = float(2*macro_avg_precision_k*marco_avg_recall_k)/(macro_avg_precision_k+marco_avg_recall_k)
            logging.info('macro_avg_precision@%d_%s: %.5f' % (topk, present_tag, macro_avg_precision_k))
            logging.info('macro_avg_recall@%d_%s: %.5f' % (topk, present_tag, marco_avg_recall_k))
            logging.info('macro_avg_f1_score@%d_%s: %.5f' % (topk, present_tag, marco_avg_f1_score_k))
            avg_dcg = sum(score_dict_all['DCG@%d_%s' % (topk, present_tag)])/len(
                score_dict_all['DCG@%d_%s' % (topk, present_tag) ])
            avg_ndcg = sum(score_dict_all['NDCG@%d_%s' % (topk, present_tag)]) / len(
                score_dict_all['NDCG@%d_%s' % (topk, present_tag)])
            avg_alpha_dcg = sum(score_dict_all['AlphaDCG@%d_%s' % (topk, present_tag)]) / len(
                score_dict_all['AlphaDCG@%d_%s' % (topk, present_tag)])
            avg_alpha_ndcg = sum(score_dict_all['AlphaNDCG@%d_%s' % (topk, present_tag)]) / len(
                score_dict_all['AlphaNDCG@%d_%s' % (topk, present_tag)])
            map = sum(score_dict_all['AP@%d_%s' % (topk, present_tag)]) / len(
                score_dict_all['AP@%d_%s' % (topk, present_tag)])
            logging.info('avg_DCG@%d_%s: %.5f' % (topk, present_tag, avg_dcg))
            logging.info('avg_NDCG@%d_%s: %.5f' % (topk, present_tag, avg_ndcg))
            logging.info('avg_Alpha_DCG@%d_%s: %.5f' % (topk, present_tag, avg_alpha_dcg))
            logging.info('avg_Alpha_NDCG@%d_%s: %.5f' % (topk, present_tag, avg_alpha_ndcg))
            logging.info('MAP@%d_%s: %.5f' % (topk, present_tag, map))
    return

if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
    description='evaluate_prediction.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.post_predict_opts(parser)
    opt = parser.parse_args()

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.filtered_pred_path = opt.filtered_pred_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.filtered_pred_path):
        os.makedirs(opt.filtered_pred_path)

    logging = config.init_logging(log_file=opt.exp_path + '/evaluate_prediction_result.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
