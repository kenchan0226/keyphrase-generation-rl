import numpy as np
import argparse
import config
from utils.string_helper import *
from collections import defaultdict
import os
import logging
import pykp.io
import pickle


def check_valid_keyphrases(str_list):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if opt.invalidate_unk:
                if w == pykp.io.UNK_WORD or w == ',' or w == '.':
                    keep_flag = False
            else:
                if w == ',' or w == '.':
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
    """
    :param keyphrase_str_list: a 2d list of tokens
    :return: a boolean np array indicate, 1 = unique, 0 = duplicate
    """
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


def check_present_keyphrases(src_str, keyphrase_str_list, match_by_str=False):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:
            if not match_by_str:  # match by word
                # check if it appears in source text
                match = False
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
            else:  # match by str
                if joined_keyphrase_str in ' '.join(src_str):
                    is_present[i] = True
                else:
                    is_present[i] = False
    return is_present


def find_present_and_absent_index(src_str, keyphrase_str_list, use_name_variations=False):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """
    num_keyphrases = len(keyphrase_str_list)
    #is_present = np.zeros(num_keyphrases, dtype=bool)
    present_indices = []
    absent_indices = []

    for i, v in enumerate(keyphrase_str_list):
        if use_name_variations:
            keyphrase_word_list = v[0]
        else:
            keyphrase_word_list = v
        joined_keyphrase_str = ' '.join(keyphrase_word_list)
        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            #is_present[i] = False
            absent_indices.append(i)
        else:
            # check if it appears in source text
            match = False
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
                #is_present[i] = True
                present_indices.append(i)
            else:
                #is_present[i] = False
                absent_indices.append(i)
    return present_indices, absent_indices


def separate_present_absent_by_source_with_variations(src_token_list, keyphrase_variation_token_3dlist, use_name_variations=True):
    num_keyphrases = len(keyphrase_variation_token_3dlist)
    present_indices = []
    absent_indices = []

    for keyphrase_idx, v in enumerate(keyphrase_variation_token_3dlist):
        if use_name_variations:
            keyphrase_variation_token_2dlist = v
        else:
            keyphrase_variation_token_2dlist = [v]
        present_flag = False
        absent_flag = False
        # iterate every variation of a keyphrase
        for variation_idx, keyphrase_token_list in enumerate(keyphrase_variation_token_2dlist):
            joined_keyphrase_str = ' '.join(keyphrase_token_list)
            if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
                absent_flag = True
            else: # check if it appears in source text
                match = False
                for src_start_idx in range(len(src_token_list) - len(keyphrase_token_list) + 1):
                    match = True
                    for keyphrase_i, keyphrase_w in enumerate(keyphrase_token_list):
                        src_w = src_token_list[src_start_idx + keyphrase_i]
                        if src_w != keyphrase_w:
                            match = False
                            break
                    if match:
                        break
                if match:
                    # is_present[i] = True
                    # present_indices.append(i)
                    present_flag = True
                else:
                    # is_present[i] = False
                    # absent_indices.append(i)
                    absent_flag = True
        if present_flag and absent_flag:
            present_indices.append(keyphrase_idx)
            absent_indices.append(keyphrase_idx)
        elif present_flag:
            present_indices.append(keyphrase_idx)
        elif absent_flag:
            absent_indices.append(keyphrase_idx)
        else:
            raise ValueError("Problem occurs in present absent checking")

    present_keyphrase_variation_token_3dlist = [keyphrase_variation_token_3dlist[present_index] for present_index in
                                                present_indices]
    absent_keyphrase_variation_token_3dlist = [keyphrase_variation_token_3dlist[absent_index] for absent_index in
                                                absent_indices]

    return present_keyphrase_variation_token_3dlist, absent_keyphrase_variation_token_3dlist


def check_present_and_duplicate_keyphrases(src_str, keyphrase_str_list, match_by_str=False):
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
        joined_keyphrase_str = ' '.join(keyphrase_word_list)
        if joined_keyphrase_str in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:
            if not match_by_str:  # match by word
                # check if it appears in source text
                match = False
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
            else:  # match by str
                if joined_keyphrase_str in ' '.join(src_str):
                    is_present[i] = True
                else:
                    is_present[i] = False
        keyphrase_set.add(joined_keyphrase_str)

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


def compute_classification_metrics_at_k(is_match, num_predictions, num_trgs, topk=5, meng_rui_precision=False):
    """
    :param is_match: a boolean np array with size [num_predictions]
    :param predicted_list:
    :param true_list:
    :param topk:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    if topk == 'M':
        topk = num_predictions
    elif topk == 'G':
        #topk = num_trgs
        if num_predictions < num_trgs:
            topk = num_trgs
        else:
            topk = num_predictions

    if meng_rui_precision:
        if num_predictions > topk:
            is_match = is_match[:topk]
            num_predictions_k = topk
        else:
            num_predictions_k = num_predictions
    else:
        if num_predictions > topk:
            is_match = is_match[:topk]
        num_predictions_k = topk

    num_matches_k = sum(is_match)

    precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_k, num_predictions_k, num_trgs)

    return precision_k, recall_k, f1_k, num_matches_k, num_predictions_k


def compute_classification_metrics_at_ks(is_match, num_predictions, num_trgs, k_list=[5,10], meng_rui_precision=False):
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
            if topk == 'M':
                topk = num_predictions
            elif topk == 'G':
                #topk = num_trgs
                if num_predictions < num_trgs:
                    topk = num_trgs
                else:
                    topk = num_predictions

            if meng_rui_precision:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk-1]
                    num_predictions_at_k = topk
                else:
                    num_matches_at_k = num_matches[-1]
                    num_predictions_at_k = num_predictions
            else:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                else:
                    num_matches_at_k = num_matches[-1]
                num_predictions_at_k = topk

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


def dcg_at_k(r, k, num_trgs, method=1):
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
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        #k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions

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


def dcg_at_ks(r, k_list, num_trgs, method=1):
    num_predictions = r.shape[0]
    if num_predictions == 0:
        dcg_array = np.array([0] * len(k_list))
    else:
        k_max = -1
        for k in k_list:
            if k == 'M':
                k = num_predictions
            elif k == 'G':
                #k = num_trgs
                if num_predictions < num_trgs:
                    k = num_trgs
                else:
                    k = num_predictions

            if k > k_max:
                k_max = k
        if num_predictions > k_max:
            r = r[:k_max]
            num_predictions = k_max
        if method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.cumsum(discounted_gain)
            return_indices = []
            for k in k_list:
                if k == 'M':
                    k = num_predictions
                elif k == 'G':
                    #k = num_trgs
                    if num_predictions < num_trgs:
                        k = num_trgs
                    else:
                        k = num_predictions

                return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
            return_indices = np.array(return_indices, dtype=int)
            dcg_array = dcg[return_indices]
        else:
            raise ValueError('method must 1.')
    return dcg_array


def ndcg_at_k(r, k, num_trgs, method=1, include_dcg=False):
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
        dcg_max = dcg_at_k(np.array(sorted(r, reverse=True)), k, num_trgs, method)
        if dcg_max <= 0.0:
            ndcg = 0.0
        else:
            dcg = dcg_at_k(r, k, num_trgs, method)
            ndcg = dcg / dcg_max
    if include_dcg:
        return ndcg, dcg
    else:
        return ndcg


def ndcg_at_ks(r, k_list, num_trgs, method=1, include_dcg=False):
    if r.shape[0] == 0:
        ndcg_array = [0.0] * len(k_list)
        dcg_array = [0.0] * len(k_list)
    else:
        dcg_array = dcg_at_ks(r, k_list, num_trgs, method)
        ideal_r = np.array(sorted(r, reverse=True))
        dcg_max_array = dcg_at_ks(ideal_r, k_list, num_trgs, method)
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
        alpha_dcg = 0.0
    else:
        # convert r_2d to gain vector
        num_trg_str, num_pred_str = r_2d.shape
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        if num_pred_str > k:
            num_pred_str = k
        gain_vector = np.zeros(num_pred_str)
        one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
        cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
        for j in range(num_pred_str):
            gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
        alpha_dcg = dcg_at_k(gain_vector, k, num_trg_str, method)
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
        return [0.0] * len(k_list)
    # convert r_2d to gain vector
    num_trg_str, num_pred_str = r_2d.shape
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str

        if k > k_max:
            k_max = k
    if num_pred_str > k_max:
        num_pred_str = k_max
    gain_vector = np.zeros(num_pred_str)
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
    for j in range(num_pred_str):
        gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
    return dcg_at_ks(gain_vector, k_list, num_trg_str, method)


def alpha_ndcg_at_k(r_2d, k, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg = 0.0
        alpha_dcg = 0.0
    else:
        num_trg_str, num_pred_str = r_2d.shape
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        # convert r to gain vector
        alpha_dcg = alpha_dcg_at_k(r_2d, k, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k, alpha)
        alpha_dcg_max = alpha_dcg_at_k(r_2d_ideal, k, method, alpha)
        if alpha_dcg_max <= 0.0:
            alpha_ndcg = 0.0
        else:
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
        # k_max = max(k_list)
        num_trg_str, num_pred_str = r_2d.shape
        k_max = -1
        for k in k_list:
            if k == 'M':
                k = num_pred_str
            elif k == 'G':
                #k = num_trg_str
                if num_pred_str < num_trg_str:
                    k = num_trg_str
                else:
                    k = num_pred_str

            if k > k_max:
                k_max = k
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
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        #k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions

    if k < num_predictions:
        num_predictions = k
        r = r[:k]
    return average_precision(r, num_predictions, num_trgs)


def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            #k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            #k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        return_indices.append( (k-1) if k <= num_predictions else (num_predictions-1) )
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]


def find_v(f1_dict, num_samples, k_list, tag):
    marco_f1_scores = np.zeros(len(k_list))
    for i, topk in enumerate(k_list):
        marco_avg_precision = f1_dict['precision_sum@{}_{}'.format(topk, tag)] / num_samples
        marco_avg_recall = f1_dict['recall_sum@{}_{}'.format(topk, tag)] / num_samples
        marco_f1_scores[i] = 2 * marco_avg_precision * marco_avg_recall / (marco_avg_precision + marco_avg_recall) if (marco_avg_precision + marco_avg_recall) > 0 else 0
        # marco_f1_scores[i] = f1_dict['f1_score_sum@{}_{}'.format(topk, tag)] / num_samples
    # for debug
    print(marco_f1_scores)
    return k_list[np.argmax(marco_f1_scores)]


def update_f1_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, f1_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)
    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list, meng_rui_precision=opt.meng_rui_precision)
    for topk, precision_k, recall_k in zip(k_list, precision_ks, recall_ks):
        f1_dict['precision_sum@{}_{}'.format(topk, tag)] += precision_k
        f1_dict['recall_sum@{}_{}'.format(topk, tag)] += recall_k
    return f1_dict


def update_f1_dict_with_name_variation(trg_variation_token_3dlist, pred_token_2dlist, k_list, f1_dict, tag):
    num_targets = len(trg_variation_token_3dlist)
    num_predictions = len(pred_token_2dlist)
    is_match = compute_var_match_result(trg_variation_token_3dlist, pred_token_2dlist)

    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=opt.meng_rui_precision)
    for topk, precision_k, recall_k in zip(k_list, precision_ks, recall_ks):
        f1_dict['precision_sum@{}_{}'.format(topk, tag)] += precision_k
        f1_dict['recall_sum@{}_{}'.format(topk, tag)] += recall_k
    return f1_dict


def update_score_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)

    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                        type='exact', dimension=1)
    is_match_substring_2d = compute_match_result(trg_token_2dlist_stemmed,
                                                 pred_token_2dlist_stemmed, type='sub', dimension=2)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list, meng_rui_precision=opt.meng_rui_precision)

    # Ranking metrics
    ndcg_ks, dcg_ks = ndcg_at_ks(is_match, k_list=k_list, num_trgs=num_targets, method=1, include_dcg=True)
    alpha_ndcg_ks, alpha_dcg_ks = alpha_ndcg_at_ks(is_match_substring_2d, k_list=k_list, method=1,
                                                   alpha=0.5, include_dcg=True)
    ap_ks = average_precision_at_ks(is_match, k_list=k_list,
                                    num_predictions=num_predictions, num_trgs=num_targets)

    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ndcg_k, dcg_k, alpha_ndcg_k, alpha_dcg_k, ap_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ndcg_ks, dcg_ks,
                alpha_ndcg_ks, alpha_dcg_ks, ap_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
        score_dict['AP@{}_{}'.format(topk, tag)].append(ap_k)
        score_dict['NDCG@{}_{}'.format(topk, tag)].append(ndcg_k)
        score_dict['AlphaNDCG@{}_{}'.format(topk, tag)].append(alpha_ndcg_k)

    score_dict['num_targets_{}'.format(tag)].append(num_targets)
    score_dict['num_predictions_{}'.format(tag)].append(num_predictions)
    return score_dict


def update_score_dict_with_name_variation_backup(is_match_all, pred_indices, num_predictions, num_targets, k_list, score_dict, tag):
    assert len(pred_indices) == num_predictions
    is_match = is_match_all[pred_indices]
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=opt.meng_rui_precision)
    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
    return score_dict


def update_score_dict_with_name_variation(trg_variation_token_3dlist, pred_token_2dlist, k_list, score_dict, tag):
    num_targets = len(trg_variation_token_3dlist)
    num_predictions = len(pred_token_2dlist)
    is_match = compute_var_match_result(trg_variation_token_3dlist, pred_token_2dlist)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=opt.meng_rui_precision)
    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
    return score_dict


def compute_var_match_result(trg_variation_token_3dlist, pred_token_2dlist):
    num_pred = len(pred_token_2dlist)
    num_trg = len(trg_variation_token_3dlist)
    is_match = np.zeros(num_pred, dtype=bool)

    for pred_idx, pred_token_list in enumerate(pred_token_2dlist):
        joined_pred_token_list = ' '.join(pred_token_list)
        match_flag = False
        for trg_idx, trg_variation_token_2dlist in enumerate(trg_variation_token_3dlist):
            for trg_variation_token_list in trg_variation_token_2dlist:
                joined_trg_variation_token_list = ' '.join(trg_variation_token_list)
                if joined_pred_token_list == joined_trg_variation_token_list:
                    is_match[pred_idx] = True
                    match_flag = True
                    break
            if match_flag:
                break
    return is_match


def filter_prediction(disable_valid_filter, disable_extra_one_word_filter, pred_token_2dlist_stemmed):
    """
    Remove the duplicate predictions, can optionally remove invalid predictions and extra one word predictions
    :param disable_valid_filter:
    :param disable_extra_one_word_filter:
    :param pred_token_2dlist_stemmed:
    :param pred_token_2d_list:
    :return:
    """
    num_predictions = len(pred_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(pred_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    pred_filter = is_unique_mask
    if not disable_valid_filter:
        is_valid_mask = check_valid_keyphrases(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * is_valid_mask
    if not disable_extra_one_word_filter:
        extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * extra_one_word_seqs_mask
    filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                          zip(pred_token_2dlist_stemmed, pred_filter) if
                                          is_keep]
    num_duplicated_predictions = num_predictions - np.sum(is_unique_mask)
    return filtered_stemmed_pred_str_list, num_duplicated_predictions


def find_unique_target(trg_token_2dlist_stemmed):
    """
    Remove the duplicate targets
    :param trg_token_2dlist_stemmed:
    :return:
    """
    num_trg = len(trg_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(trg_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    trg_filter = is_unique_mask
    filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in
                                      zip(trg_token_2dlist_stemmed, trg_filter) if
                                      is_keep]
    num_duplicated_trg = num_trg - np.sum(is_unique_mask)
    return filtered_stemmed_trg_str_list, num_duplicated_trg


def separate_present_absent_by_source(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str):
    is_present_mask = check_present_keyphrases(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str)
    present_keyphrase_token2dlist = []
    absent_keyphrase_token2dlist = []
    for keyphrase_token_list, is_present in zip(keyphrase_token_2dlist_stemmed, is_present_mask):
        if is_present:
            present_keyphrase_token2dlist.append(keyphrase_token_list)
        else:
            absent_keyphrase_token2dlist.append(keyphrase_token_list)
    return present_keyphrase_token2dlist, absent_keyphrase_token2dlist


def separate_present_absent_by_segmenter(keyphrase_token_2dlist, segmenter):
    present_keyphrase_token2dlist = []
    absent_keyphrase_token2dlist = []
    absent_flag = False
    for keyphrase_token_list in keyphrase_token_2dlist:
        if keyphrase_token_list[0] == segmenter:
            absent_flag = True
            # skip the segmenter token, because it should not be included in the evaluation
            continue
        if absent_flag:
            absent_keyphrase_token2dlist.append(keyphrase_token_list)
        else:
            present_keyphrase_token2dlist.append(keyphrase_token_list)
    return present_keyphrase_token2dlist, absent_keyphrase_token2dlist


def process_input_ks(ks):
    ks_list = []
    for k in ks:
        if k != 'M' and k != 'G':
            k = int(k)
        ks_list.append(k)
    return ks_list


def main(opt):
    src_file_path = opt.src_file_path
    trg_file_path = opt.trg_file_path
    pred_file_path = opt.pred_file_path

    if opt.export_filtered_pred:
        pred_output_file = open(os.path.join(opt.filtered_pred_path, "predictions_filtered.txt"), "w")

    if opt.tune_f1_v:
        f1_dict = defaultdict(lambda: 0)
        max_k = 20
        topk_dict = {'present': list(range(1, max_k)), 'absent': list(range(1, max_k)), 'all': list(range(1, max_k))}
    else:
        score_dict = defaultdict(list)
        all_ks = process_input_ks(opt.all_ks)
        present_ks = process_input_ks(opt.present_ks)
        absent_ks = process_input_ks(opt.absent_ks)
        topk_dict = {'present': present_ks, 'absent': absent_ks, 'all': all_ks}
        # topk_dict = {'present': [5, 10, 'M'], 'absent': [5, 10, 50, 'M'], 'all': [5, 10, 'M']}

    total_num_src = 0
    total_num_src_with_present_keyphrases = 0
    total_num_src_with_absent_keyphrases = 0
    total_num_unique_predictions = 0
    total_num_present_filtered_predictions = 0
    total_num_present_unique_targets = 0
    total_num_absent_filtered_predictions = 0
    total_num_absent_unique_targets = 0
    max_unique_targets = 0

    if opt.prediction_separated:
        sum_incorrect_fraction_for_identifying_present = 0
        sum_incorrect_fraction_for_identifying_absent = 0

    for data_idx, (src_l, trg_l, pred_l) in enumerate(zip(open(src_file_path), open(trg_file_path), open(pred_file_path))):
        total_num_src += 1
        # convert the str to token list
        pred_str_list = pred_l.strip().split(';')
        pred_str_list = pred_str_list[:opt.num_preds]
        pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred_str_list]
        trg_str_list = trg_l.strip().split(';')
        if opt.use_name_variations:
            #trg_token_2dlist = [trg_str.strip().split('|') for trg_str in trg_str_list]
            trg_variation_token_3dlist = []
            for trg_str in trg_str_list:
                name_variation_list = trg_str.strip().split('|')
                name_variation_tokens_2dlist = []
                for name_variation in name_variation_list:
                    name_variation_tokens_2dlist.append(name_variation.strip().split())
                trg_variation_token_3dlist.append(name_variation_tokens_2dlist)
        else:
            trg_token_2dlist = [trg_str.strip().split(' ') for trg_str in trg_str_list]

        # TODO: test name_variation_tokens_3dlist
        [title, context] = src_l.strip().split('<eos>')
        src_token_list = title.strip().split(' ') + context.strip().split(' ')

        num_predictions = len(pred_str_list)

        # perform stemming
        stemmed_src_token_list = stem_word_list(src_token_list)

        if opt.use_name_variations:
            if opt.target_already_stemmed:
                stemmed_trg_variation_token_3dlist = trg_variation_token_3dlist
            else:
                stemmed_trg_variation_token_3dlist = stem_str_2d_list(trg_variation_token_3dlist)
        else:
            if opt.target_already_stemmed:
                stemmed_trg_token_2dlist = trg_token_2dlist
            else:
                stemmed_trg_token_2dlist = stem_str_list(trg_token_2dlist)
        # TODO: test stemmed_trg_variation_token_3dlist

        stemmed_pred_token_2dlist = stem_str_list(pred_token_2dlist)

        # remove peos in predictions, then check if the model can successfuly separate present and absent keyphrases by segmenter
        if opt.prediction_separated:
            if opt.reverse_sorting:
                absent_stemmed_pred_token_2dlist_by_segmenter, present_stemmed_pred_token_2dlist_by_segmenter = separate_present_absent_by_segmenter(
                    stemmed_pred_token_2dlist, present_absent_segmenter)
            else:
                present_stemmed_pred_token_2dlist_by_segmenter, absent_stemmed_pred_token_2dlist_by_segmenter = separate_present_absent_by_segmenter(stemmed_pred_token_2dlist, present_absent_segmenter)
            stemmed_pred_token_2dlist = present_stemmed_pred_token_2dlist_by_segmenter + absent_stemmed_pred_token_2dlist_by_segmenter  # remove all the peos token
            # check present absent
            num_absent_before_segmenter = len(present_stemmed_pred_token_2dlist_by_segmenter) - sum(check_present_keyphrases(stemmed_src_token_list, present_stemmed_pred_token_2dlist_by_segmenter))
            num_present_after_segmenter = sum(check_present_keyphrases(stemmed_src_token_list, absent_stemmed_pred_token_2dlist_by_segmenter))
            incorrect_fraction_for_identifying_present = num_absent_before_segmenter/len(present_stemmed_pred_token_2dlist_by_segmenter) if len(present_stemmed_pred_token_2dlist_by_segmenter) > 0 else 0
            incorrect_fraction_for_identifying_absent = num_present_after_segmenter/len(absent_stemmed_pred_token_2dlist_by_segmenter) if len(absent_stemmed_pred_token_2dlist_by_segmenter) > 0 else 0
            sum_incorrect_fraction_for_identifying_present += incorrect_fraction_for_identifying_present
            sum_incorrect_fraction_for_identifying_absent += incorrect_fraction_for_identifying_absent

        # Filter out duplicate, invalid, and extra one word predictions
        filtered_stemmed_pred_token_2dlist, num_duplicated_predictions = filter_prediction(opt.disable_valid_filter, opt.disable_extra_one_word_filter, stemmed_pred_token_2dlist)
        total_num_unique_predictions += (num_predictions - num_duplicated_predictions)
        num_filtered_predictions = len(filtered_stemmed_pred_token_2dlist)

        # Remove duplicated targets
        if opt.use_name_variations:  # testing set with name variation have removed all duplicates during preprocessing
            num_unique_targets = len(stemmed_trg_variation_token_3dlist)
            unique_stemmed_trg_variation_token_3dlist = stemmed_trg_variation_token_3dlist
        else:
            unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)
            #unique_stemmed_trg_token_2dlist = stemmed_trg_token_2dlist
            num_unique_targets = len(unique_stemmed_trg_token_2dlist)
            # max_unique_targets += (num_trg - num_duplicated_trg)

        if num_unique_targets > max_unique_targets:
            max_unique_targets = num_unique_targets

        # separate present and absent keyphrases
        if opt.use_name_variations:
            # separate prediction
            present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist = \
                separate_present_absent_by_source_with_variations(stemmed_src_token_list,
                                                              filtered_stemmed_pred_token_2dlist,
                                                              use_name_variations=False)
            # separate target
            present_unique_stemmed_trg_variation_token_3dlist, absent_unique_stemmed_trg_variation_token_3dlist = \
            separate_present_absent_by_source_with_variations(stemmed_src_token_list,
                                                              unique_stemmed_trg_variation_token_3dlist,
                                                              use_name_variations=True)

            num_present_filtered_predictions = len(present_filtered_stemmed_pred_token_2dlist)
            num_present_unique_targets = len(present_unique_stemmed_trg_variation_token_3dlist)
            num_absent_filtered_predictions = len(absent_filtered_stemmed_pred_token_2dlist)
            num_absent_unique_targets = len(absent_unique_stemmed_trg_variation_token_3dlist)

            total_num_present_filtered_predictions += num_present_filtered_predictions
            total_num_present_unique_targets += num_present_unique_targets
            total_num_absent_filtered_predictions += num_absent_filtered_predictions
            total_num_absent_unique_targets += num_absent_unique_targets

            if num_present_unique_targets > 0:
                total_num_src_with_present_keyphrases += 1
            if num_absent_unique_targets > 0:
                total_num_src_with_absent_keyphrases += 1

            if opt.tune_f1_v:
                # compute F1 score all
                f1_dict = update_f1_dict(unique_stemmed_trg_variation_token_3dlist, filtered_stemmed_pred_token_2dlist,
                                         topk_dict['all'], f1_dict, 'all')
                # compute F1 score present
                f1_dict = update_f1_dict(present_unique_stemmed_trg_variation_token_3dlist,
                                         present_filtered_stemmed_pred_token_2dlist,
                                         topk_dict['present'], f1_dict, 'present')
                # compute F1 score absent
                f1_dict = update_f1_dict(absent_unique_stemmed_trg_variation_token_3dlist,
                                         absent_filtered_stemmed_pred_token_2dlist,
                                         topk_dict['absent'], f1_dict, 'absent')
            else:
                # compute all the metrics and update the score_dict
                score_dict = update_score_dict_with_name_variation(unique_stemmed_trg_variation_token_3dlist,
                                                                   filtered_stemmed_pred_token_2dlist,
                                                      topk_dict['all'], score_dict, 'all')
                # compute all the metrics and update the score_dict for present keyphrase
                score_dict = update_score_dict_with_name_variation(present_unique_stemmed_trg_variation_token_3dlist,
                                                                   present_filtered_stemmed_pred_token_2dlist,
                                                                   topk_dict['present'], score_dict, 'present')
                # compute all the metrics and update the score_dict for present keyphrase
                score_dict = update_score_dict_with_name_variation(absent_unique_stemmed_trg_variation_token_3dlist,
                                                                   absent_filtered_stemmed_pred_token_2dlist,
                                                                   topk_dict['absent'], score_dict, 'absent')

        else:
            present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist = separate_present_absent_by_source(stemmed_src_token_list, filtered_stemmed_pred_token_2dlist, opt.match_by_str)
            if opt.target_separated:
                if opt.reverse_sorting:
                    absent_unique_stemmed_trg_token_2dlist, present_unique_stemmed_trg_token_2dlist = separate_present_absent_by_segmenter(
                        unique_stemmed_trg_token_2dlist, present_absent_segmenter)
                else:
                    present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist = separate_present_absent_by_segmenter(
                        unique_stemmed_trg_token_2dlist, present_absent_segmenter)
            else:
                present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist = separate_present_absent_by_source(
                    stemmed_src_token_list, unique_stemmed_trg_token_2dlist, opt.match_by_str)

            total_num_present_filtered_predictions += len(present_filtered_stemmed_pred_token_2dlist)
            total_num_present_unique_targets += len(present_unique_stemmed_trg_token_2dlist)
            total_num_absent_filtered_predictions += len(absent_filtered_stemmed_pred_token_2dlist)
            total_num_absent_unique_targets += len(absent_unique_stemmed_trg_token_2dlist)
            if len(present_unique_stemmed_trg_token_2dlist) > 0:
                total_num_src_with_present_keyphrases += 1
            if len(absent_unique_stemmed_trg_token_2dlist) > 0:
                total_num_src_with_absent_keyphrases += 1

            if opt.tune_f1_v:
                # compute F1 score all
                f1_dict = update_f1_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                         topk_dict['all'], f1_dict, 'all')
                # compute F1 score present
                f1_dict = update_f1_dict(present_unique_stemmed_trg_token_2dlist,
                                         present_filtered_stemmed_pred_token_2dlist,
                                         topk_dict['present'], f1_dict, 'present')
                # compute F1 score absent
                f1_dict = update_f1_dict(absent_unique_stemmed_trg_token_2dlist,
                                         absent_filtered_stemmed_pred_token_2dlist,
                                         topk_dict['absent'], f1_dict, 'absent')
            else:
                # compute all the metrics and update the score_dict
                score_dict = update_score_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                               topk_dict['all'], score_dict, 'all')
                # compute all the metrics and update the score_dict for present keyphrase
                score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist,
                                               present_filtered_stemmed_pred_token_2dlist,
                                               topk_dict['present'], score_dict, 'present')
                # compute all the metrics and update the score_dict for present keyphrase
                score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist,
                                               absent_filtered_stemmed_pred_token_2dlist,
                                               topk_dict['absent'], score_dict, 'absent')
        """
        if opt.tune_f1_v:
            # compute F1 score all
            f1_dict = update_f1_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                           topk_dict['all'], f1_dict, 'all')
            # compute F1 score present
            f1_dict = update_f1_dict(present_unique_stemmed_trg_token_2dlist, present_filtered_stemmed_pred_token_2dlist,
                                           topk_dict['present'], f1_dict, 'present')
            # compute F1 score absent
            f1_dict = update_f1_dict(absent_unique_stemmed_trg_token_2dlist, absent_filtered_stemmed_pred_token_2dlist,
                                           topk_dict['absent'], f1_dict, 'absent')
        else:
            # compute all the metrics and update the score_dict
            score_dict = update_score_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                           topk_dict['all'], score_dict, 'all')
            # compute all the metrics and update the score_dict for present keyphrase
            score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist, present_filtered_stemmed_pred_token_2dlist,
                                           topk_dict['present'], score_dict, 'present')
            # compute all the metrics and update the score_dict for present keyphrase
            score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist, absent_filtered_stemmed_pred_token_2dlist,
                                           topk_dict['absent'], score_dict, 'absent')
        """
        if opt.export_filtered_pred:
            final_pred_str_list = []
            for word_list in filtered_stemmed_pred_token_2dlist:
                final_pred_str_list.append(' '.join(word_list))
            pred_print_out = ';'.join(final_pred_str_list) + '\n'
            pred_output_file.write(pred_print_out)

    if opt.export_filtered_pred:
        pred_output_file.close()

    if opt.tune_f1_v:
        v_all = find_v(f1_dict, total_num_src, topk_dict['all'], 'all')
        print("V for all {}".format(v_all))
        v_present = find_v(f1_dict, total_num_src, topk_dict['present'], 'present')
        print("V for present {}".format(v_present))
        v_absent = find_v(f1_dict, total_num_src, topk_dict['absent'], 'absent')
        print("V for absent {}".format(v_absent))
        v_file = open(os.path.join(opt.exp_path, "tune_v_output.txt"), "w")
        v_file.write("V for all {}\n".format(v_all))
        v_file.write("V for present {}\n".format(v_present))
        v_file.write("V for absent {}\n".format(v_absent))
        v_file.close()
    else:
        total_num_unique_targets = total_num_present_unique_targets + total_num_absent_unique_targets
        total_num_filtered_predictions = total_num_present_filtered_predictions + total_num_absent_filtered_predictions

        result_txt_str = ""

        # report global statistics
        result_txt_str += ('Total #samples: %d\t # samples with present keyphrases: %d\t # samples with absent keyphrases: %d\n' % (total_num_src, total_num_src_with_present_keyphrases, total_num_src_with_absent_keyphrases))
        result_txt_str += ('Max. unique targets per src: %d\n' % (max_unique_targets))
        result_txt_str += ('Total #unique predictions: %d\n' % total_num_unique_predictions)

        # report statistics and scores for all predictions and targets
        result_txt_str_all, field_list_all, result_list_all = report_stat_and_scores(total_num_filtered_predictions, total_num_unique_targets, total_num_src, score_dict, topk_dict['all'], 'all', opt.use_name_variations)
        result_txt_str_present, field_list_present, result_list_present = report_stat_and_scores(total_num_present_filtered_predictions, total_num_present_unique_targets, total_num_src, score_dict, topk_dict['present'], 'present', opt.use_name_variations)
        result_txt_str_absent, field_list_absent, result_list_absent = report_stat_and_scores(total_num_absent_filtered_predictions, total_num_absent_unique_targets, total_num_src, score_dict, topk_dict['absent'], 'absent', opt.use_name_variations)
        result_txt_str += (result_txt_str_all + result_txt_str_present + result_txt_str_absent)
        field_list = field_list_all + field_list_present + field_list_absent
        result_list = result_list_all + result_list_present + result_list_absent

        # Write to files
        # topk_dict = {'present': [5, 10, 'M'], 'absent': [5, 10, 50, 'M'], 'all': [5, 10, 'M']}
        k_list = topk_dict['all'] + topk_dict['present'] + topk_dict['absent']
        result_file_suffix = '_'.join([str(k) for k in k_list])
        if opt.meng_rui_precision:
            result_file_suffix += '_meng_rui_precision'
        if opt.use_name_variations:
            result_file_suffix += '_name_variations'
        results_txt_file = open(os.path.join(opt.exp_path, "results_log_{}.txt".format(result_file_suffix)), "w")
        if opt.prediction_separated:
            result_txt_str += "===================================Separation====================================\n"
            result_txt_str += "Avg error fraction for identifying present keyphrases: {:.5}\n".format(sum_incorrect_fraction_for_identifying_present / total_num_src)
            result_txt_str += "Avg error fraction for identifying absent keyphrases: {:.5}\n".format(sum_incorrect_fraction_for_identifying_absent / total_num_src)

        # Report MAE on lengths
        result_txt_str += "===================================MAE stat====================================\n"

        num_targets_present_array = np.array(score_dict['num_targets_present'])
        num_predictions_present_array = np.array(score_dict['num_predictions_present'])
        num_targets_absent_array = np.array(score_dict['num_targets_absent'])
        num_predictions_absent_array = np.array(score_dict['num_predictions_absent'])

        all_mae = mae(num_targets_present_array+num_targets_absent_array, num_predictions_present_array+num_predictions_absent_array)
        present_mae = mae(num_targets_present_array, num_predictions_present_array)
        absent_mae = mae(num_targets_absent_array, num_predictions_absent_array)

        result_txt_str += "MAE on keyphrase numbers (all): {:.5}\n".format(all_mae)
        result_txt_str += "MAE on keyphrase numbers (present): {:.5}\n".format(present_mae)
        result_txt_str += "MAE on keyphrase numbers (absent): {:.5}\n".format(absent_mae)

        results_txt_file.write(result_txt_str)
        results_txt_file.close()

        results_tsv_file = open(os.path.join(opt.exp_path, "results_log_{}.tsv".format(result_file_suffix)), "w")
        results_tsv_file.write('\t'.join(field_list) + '\n')
        results_tsv_file.write('\t'.join('%.5f' % result for result in result_list) + '\n')
        results_tsv_file.close()

        # save score dict for future use
        score_dict_pickle = open(os.path.join(opt.exp_path, "score_dict_{}.pickle".format(result_file_suffix)), "wb")
        pickle.dump(score_dict, score_dict_pickle)
        score_dict_pickle.close()

    return


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def mae(a, b):
    return (np.abs(a - b)).mean()


def report_stat_and_scores(total_num_filtered_predictions, num_unique_trgs, total_num_src, score_dict, topk_list, present_tag, use_name_variations=False):
    result_txt_str = "===================================%s====================================\n" % (present_tag)
    result_txt_str += "#predictions after filtering: %d\t #predictions after filtering per src:%.3f\n" % \
                      (total_num_filtered_predictions, total_num_filtered_predictions / total_num_src)
    result_txt_str += "#unique targets: %d\t #unique targets per src:%.3f\n" % \
                      (num_unique_trgs, num_unique_trgs / total_num_src)

    classification_output_str, classification_field_list, classification_result_list = report_classification_scores(
        score_dict, topk_list, present_tag)
    result_txt_str += classification_output_str
    field_list = classification_field_list
    result_list = classification_result_list

    if not use_name_variations:
        ranking_output_str, ranking_field_list, ranking_result_list = report_ranking_scores(score_dict,
                                                                                            topk_list,
                                                                                            present_tag)
        result_txt_str += ranking_output_str
        field_list += ranking_field_list
        result_list += ranking_result_list
    return result_txt_str, field_list, result_list


def report_classification_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []
    for topk in topk_list:
        total_predictions_k = sum(score_dict['num_predictions@{}_{}'.format(topk, present_tag)])
        total_targets_k = sum(score_dict['num_targets@{}_{}'.format(topk, present_tag)])
        total_num_matches_k = sum(score_dict['num_matches@{}_{}'.format(topk, present_tag)])
        # Compute the micro averaged recall, precision and F-1 score
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(
            total_num_matches_k, total_predictions_k, total_targets_k)
        # Compute the macro averaged recall, precision and F-1 score
        macro_avg_precision_k = sum(score_dict['precision@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['precision@{}_{}'.format(topk, present_tag)])
        macro_avg_recall_k = sum(score_dict['recall@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['recall@{}_{}'.format(topk, present_tag)])
        macro_avg_f1_score_k = (2 * macro_avg_precision_k * macro_avg_recall_k) / (macro_avg_precision_k + macro_avg_recall_k) if (macro_avg_precision_k + macro_avg_recall_k) > 0 else 0.0
        output_str += ("Begin===============classification metrics {}@{}===============Begin\n".format(present_tag, topk))
        output_str += ("#target: {}, #predictions: {}, #corrects: {}\n".format(total_predictions_k, total_targets_k, total_num_matches_k))
        output_str += "Micro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, micro_avg_precision_k, topk, micro_avg_recall_k, topk, micro_avg_f1_score_k)
        output_str += "Macro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, macro_avg_precision_k, topk, macro_avg_recall_k, topk, macro_avg_f1_score_k)
        field_list += ['macro_avg_p@{}_{}'.format(topk, present_tag), 'macro_avg_r@{}_{}'.format(topk, present_tag), 'macro_avg_f1@{}_{}'.format(topk, present_tag)]
        result_list += [macro_avg_precision_k, macro_avg_recall_k, macro_avg_f1_score_k]
    return output_str, field_list, result_list


def report_ranking_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []
    for topk in topk_list:
        map_k = sum(score_dict['AP@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['AP@{}_{}'.format(topk, present_tag)])
        #avg_dcg_k = sum(score_dict['DCG@{}_{}'.format(topk, present_tag)]) / len(
        #    score_dict['DCG@{}_{}'.format(topk, present_tag)])
        avg_ndcg_k = sum(score_dict['NDCG@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['NDCG@{}_{}'.format(topk, present_tag)])
        #avg_alpha_dcg_k = sum(score_dict['AlphaDCG@{}_{}'.format(topk, present_tag)]) / len(
        #    score_dict['AlphaDCG@{}_{}'.format(topk, present_tag)])
        avg_alpha_ndcg_k = sum(score_dict['AlphaNDCG@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['AlphaNDCG@{}_{}'.format(topk, present_tag)])
        output_str += ("Begin==================Ranking metrics {}@{}==================Begin\n".format(present_tag, topk))
        #output_str += "Relevance:\tDCG@{}={:.5}\tNDCG@{}={:.5}\tMAP@{}={:.5}\n".format(topk, avg_dcg_k, topk, avg_ndcg_k, topk, map_k)
        #output_str += "Diversity:\tAlphaDCG@{}={:.5}\tAlphaNDCG@{}={:.5}\n".format(topk, avg_alpha_dcg_k, topk, avg_alpha_ndcg_k)
        #field_list += ['avg_DCG@{}_{}'.format(topk, present_tag), 'avg_NDCG@{}_{}'.format(topk, present_tag),
        #               'MAP@{}_{}'.format(topk, present_tag), 'AlphaDCG@{}_{}'.format(topk, present_tag), 'AlphaNDCG@{}_{}'.format(topk, present_tag)]
        #result_list += [avg_dcg_k, avg_ndcg_k, map_k, avg_alpha_dcg_k, avg_alpha_ndcg_k]
        output_str += "\tMAP@{}={:.5}\tNDCG@{}={:.5}\tAlphaNDCG@{}={:.5}\n".format(topk, map_k, topk, avg_ndcg_k, topk, avg_alpha_ndcg_k)
        field_list += ['MAP@{}_{}'.format(topk, present_tag), 'avg_NDCG@{}_{}'.format(topk, present_tag), 'AlphaNDCG@{}_{}'.format(topk, present_tag)]
        result_list += [map_k, avg_ndcg_k, avg_alpha_ndcg_k]

    return output_str, field_list, result_list


if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
    description='evaluate_prediction.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.post_predict_opts(parser)
    opt = parser.parse_args()

    if opt.exp_path == "" and opt.filtered_pred_path == "":
        pred_folder_path = os.path.split(opt.pred_file_path)[0]
        exp_folder_path = pred_folder_path.replace("pred/", "exp/")
        opt.exp_path = exp_folder_path
        opt.filtered_pred_path = pred_folder_path

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.filtered_pred_path = opt.filtered_pred_path % (opt.exp, opt.timemark)

    present_absent_segmenter = '<peos>'

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.filtered_pred_path):
        os.makedirs(opt.filtered_pred_path)

    logging = config.init_logging(log_file=opt.exp_path + '/evaluate_prediction_result.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
