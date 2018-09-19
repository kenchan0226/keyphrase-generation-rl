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


def compute_match_result(trg_str_list, pred_str_list, type='exact'):
    assert type in ['exact'], "Right now only support exact matching"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    is_match = np.zeros(num_pred_str, dtype=bool)

    for pred_idx, pred_word_list in enumerate(pred_str_list):
        if type == 'exact':
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
        num_predictions = topk

    num_matches = sum(is_match)

    precision_k, recall_k, f1_k = compute_classificatioon_metrics(num_matches, num_predictions, num_trgs)

    return precision_k, recall_k, f1_k, num_matches, num_predictions


def compute_classificatioon_metrics(num_matches, num_predictions, num_trgs):
    precision = num_matches / num_predictions if num_predictions > 0 else 0.0
    recall = num_matches / num_trgs if num_trgs > 0 else 0.0

    if precision + recall > 0:
        f1 = float(2 * (precision * recall)) / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def dcg_at_k(r, k, method=0):
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
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
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
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def main(opt):
    src_file_path = opt.src_file_path
    trg_file_path = opt.trg_file_path
    pred_file_path = opt.pred_file_path

    if opt.export_filtered_pred:
        pred_output_file = open(os.path.join(opt.filtered_pred_path, "predictions_filtered.txt"), "w")

    # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    score_dict_all = defaultdict(list)
    topk_dict = {'present': [5, 10], 'absent': [10, 50]}

    num_src = 0
    num_unique_predictions = 0
    num_unique_present_predictions = 0
    num_unique_absent_predictions = 0
    num_unique_present_filtered_predictions = 0
    num_unique_present_filtered_targets = 0
    num_unique_absent_filtered_predictions = 0
    num_unique_absent_filtered_targets = 0
    max_unique_targets = 0

    for src_l, trg_l, pred_l in zip(open(src_file_path), open(trg_file_path), open(pred_file_path)):
        num_src += 1
        pred_str_list = pred_l.strip().split(';')
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
            is_match = compute_match_result(trg_str_list=filtered_stemmed_trg_str_list,
                                            pred_str_list=filtered_stemmed_pred_str_list)

            num_filtered_predictions = len(filtered_pred_str_list)
            num_filtered_targets = len(filtered_trg_str_list)
            if is_present:
                num_unique_present_filtered_predictions += num_filtered_predictions
                num_unique_present_filtered_targets += num_filtered_targets
            else:
                num_unique_absent_filtered_predictions += num_filtered_predictions
                num_unique_absent_filtered_targets += num_filtered_targets

            for topk in topk_dict[present_tag]:
                precision_k, recall_k, f1_k, num_matches_k, num_predictions_k = \
                    compute_classification_metrics_at_k(is_match, num_filtered_predictions, num_filtered_targets,
                                                        topk=topk)
                results = prepare_classification_result_dict(precision_k, recall_k, f1_k, num_matches_k,
                                                             num_predictions_k, num_filtered_targets, topk, is_present)
                for metric, result in results.items():
                    score_dict_all[metric].append(result)

                #print("Result of %s@%d" % (present_tag, topk))
                #print(results)


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
            micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classificatioon_metrics(sum(score_dict_all['num_matches@%d_%s' % (topk, present_tag)]), total_predictions, total_targets)
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
