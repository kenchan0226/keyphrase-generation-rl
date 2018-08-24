#from nltk.stem.porter import *
import torch
#from utils import Progbar
#from pykp.metric.bleu import bleu
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import Statistics
import time
from utils.time_log import time_since
from nltk.stem.porter import *
import pykp
import logging
import numpy as np
from collections import defaultdict

stemmer = PorterStemmer()

def process_predseqs(pred_seq_list, oov, idx2word, opt):
    '''
    :param pred_seqs: a list of sequence objects
    :param oov: a list of oov words
    :param idx2word: a dictionary
    :param opt:
    :return:
    '''
    processed_seq_list = []
    num_pred_seq = len(pred_seq_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)

    for seq_i, seq in enumerate(pred_seq_list):
        # convert to words and remove the EOS token. Each idx in seq.word_idx_list is a 0 dimension tensor, need to convert to python int first
        word_list = [idx2word[int(x.item())] if x < opt.vocab_size else oov[int(x.item()) - opt.vocab_size] for x in seq.word_idx_list[:-1]]

        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if w == pykp.io.UNK_WORD or w == ',' or w == '.':
                keep_flag = False

        is_valid[seq_i] = keep_flag
        processed_seq_list.append((seq, word_list, seq.score))

    unzipped = list(zip(*(processed_seq_list)))
    processed_seq_list, processed_str_lists, processed_scores = unzipped if len(processed_seq_list) > 0 and len(unzipped) == 3 else ([], [], [])

    assert len(processed_seq_list) == len(processed_str_lists) == len(processed_scores) == len(is_valid)
    return is_valid, processed_seq_list, processed_str_lists, processed_scores

def filter_one_word_sequences(str_list):
    """
    only keep the first one-word prediction, remove all the remaining keyphrases that only has one word.
    :param str_list: a list of word lists
    :return: filtered_string_list, num_one_word_seqs
    """
    filtered_string_list = []
    num_one_word_seqs = 0
    for word_list in str_list:
        if len(word_list) == 1:
            num_one_word_seqs += 1
            if num_one_word_seqs > 1:
                continue
        filtered_string_list.append(word_list)

    return filtered_string_list, num_one_word_seqs

def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]

def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    total_batch = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            total_batch += 1
            src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists = batch

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            start_time = time.time()
            decoder_dist, h_t, attention_dist, coverage = model(src, src_lens, trg, src_oov, max_num_oov, src_mask)
            forward_time = time_since(start_time)

            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage)
            loss_compute_time = time_since(start_time)

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

    eval_loss_stat = Statistics(evaluation_loss_sum, total_trg_tokens, total_batch, forward_time=forward_time, loss_compute_time=loss_compute_time)

    return eval_loss_stat


def check_present_and_duplicate_keyphrases(src_str, keyphrase_str_list):
    stemmed_src_str = stem_word_list(src_str)
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    stemmed_keyphrase_str_list = []
    stemmed_keyphrase_set = set()

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        stemmed_keyphrase_word_list = stem_word_list(keyphrase_word_list)
        stemmed_keyphrase_str_list.append(stemmed_keyphrase_word_list)
        if '_'.join(stemmed_keyphrase_word_list) in stemmed_keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True

        # check if it appears in source text
        for src_start_idx in range(len(stemmed_src_str) - len(stemmed_keyphrase_word_list) + 1):
            match = True
            for keyphrase_i, keyphrase_w in enumerate(stemmed_keyphrase_word_list):
                src_w = stemmed_src_str[src_start_idx + keyphrase_i]
                if src_w != keyphrase_w:
                    match = False
                    break
            if match:
                break

        if match:
            is_present[i] = True
        else:
            is_present[i] = False
        stemmed_keyphrase_set.add('_'.join(stemmed_keyphrase_word_list))

    return is_present, not_duplicate, stemmed_keyphrase_str_list


'''
def if_present_duplicate_phrase(src_str, phrase_seqs):
    stemmed_src_str = stem_word_list(src_str)
    present_index = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for phrase_seq in phrase_seqs:
        stemmed_pred_seq = stem_word_list(phrase_seq)

        # check if it is duplicate
        if '_'.join(stemmed_pred_seq) in phrase_set:
            present_index.append(False)
            continue

        # check if it appears in source text
        for src_start_idx in range(len(stemmed_src_str) - len(stemmed_pred_seq) + 1):
            match = True
            for seq_idx, seq_w in enumerate(stemmed_pred_seq):
                src_w = stemmed_src_str[src_start_idx + seq_idx]
                if src_w != seq_w:
                    match = False
                    break
            if match:
                break

        # if it reaches the end of source and no match, means it doesn't appear in the source, thus discard
        if match:
            present_index.append(True)
        else:
            present_index.append(False)
        phrase_set.add('_'.join(stemmed_pred_seq))

    return present_index
'''


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

    return {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1_k, 'num_matches@%d': num_matches}


def compute_classificatioon_metrics(num_matches, num_predictions, num_trgs):
    precision = num_matches / num_predictions if num_predictions > 0 else 0.0
    recall = num_matches / num_trgs if num_trgs > 0 else 0.0

    if precision + recall > 0:
        f1 = float(2 * (precision * recall)) / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def evaluate_beam_search(generator, one2many_data_loader, opt, save_path=None):
    score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    example_idx = 0
    total_predictions = 0
    total_targets = 0

    for batch_i, batch in enumerate(one2many_data_loader):
        src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist = batch
        """
        src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        src_lens: a list containing the length of src sequences for each batch, with len=batch
        src_mask: a FloatTensor, [batch, src_seq_len]
        src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        oov_lists: a list of oov words for each src, 2dlist
        """
        pred_seq_2dlist = generator.beam_search(src, src_lens, src_oov, oov_lists, opt.word2idx)
        # a list of list of sequence objs, len(pred_seq_2dlist)=batch_size, len(pred_seq_2dlist)=beam_size

        # Process it every src in the batch
        for src_str, trg_str_list, pred_seq_list, oov in zip(src_str_list, trg_str_2dlist, pred_seq_2dlist, oov_lists):
            # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
            # pred_seq_list: a list of sequence objects, sorted by scores
            # oov: a list of oov words
            print_out = ''
            print_out += '[Source][%d]: %s \n' % (len(src_str), ' '.join(src_str))

            # is_present: boolean np array indicate whether a predicted keyphrase is present in src
            # not_duplicate: boolean np array indicate
            trg_str_is_present, trg_str_not_duplicate, stemmed_trg_str_list = check_present_and_duplicate_keyphrases(src_str, trg_str_list)

            print_out += '[GROUND-TRUTH] #(present)/#(all targets)=%d/%d\n' % (sum(trg_str_is_present), len(trg_str_list))
            print_out += '\n'.join(
                ['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in
                 zip(trg_str_list, trg_str_is_present)])
            print_out += '\noov_list:   \n\t\t%s \n' % str(oov)

            # convert each idx in pred_seq to its word, processed_pred_str_list: a list of word list
            # a pred_seq is invalid if len(processed_seq) == 0 or keep_flag and any word in processed_seq is UNK or it contains '.' or ','
            pred_str_is_valid, processed_pred_seq_list, processed_pred_str_list, processed_pred_score_list = process_predseqs(pred_seq_list, oov, opt.idx2word, opt)

            # a list of boolean indicates which predicted keyphrases present in src, for the duplicated keyphrases after stemming, only consider the first one
            pred_str_is_present, pred_str_not_duplicate, stemmed_pred_str_list = check_present_and_duplicate_keyphrases(src_str, processed_pred_str_list)

            # print out basic stats of predicted keyphrases and each keyphrase
            print_out += '[PREDICTION] #(valid)=%d, #(present)=%d, #(all)=%d\n' % (
                sum(pred_str_is_valid), sum(pred_str_is_present), len(pred_seq_list))
            for word_list, is_present in zip(processed_pred_str_list, pred_str_is_present):
                if is_present:
                    print_out += '\t\t[%s]' % ' '.join(word_list)
                else:
                    print_out += '\t\t%s' % ' '.join(word_list)
            print_out += '\n'


            trg_str_filter = trg_str_not_duplicate  # filter the duplicated target keyphrases
            pred_str_filter = pred_str_not_duplicate * pred_str_is_valid # filter the duplicated and invalid predicted keyphrases
            if opt.present_kp_only: # only keep the keyphrases that present in src
                trg_str_filter = trg_str_filter * trg_str_is_present
                pred_str_filter = pred_str_filter * pred_str_is_present

            # Apply filter
            trg_str_list = [word_list for word_list, is_keep in zip(trg_str_list, trg_str_filter) if
                            is_keep]
            #processed_pred_seq_list = [seq for seq, is_keep in zip(processed_pred_seq_list, pred_str_filter) if is_keep]
            processed_pred_str_list = [word_list for word_list, is_keep in zip(processed_pred_str_list, pred_str_filter) if
                                       is_keep]
            #processed_pred_score_list = [score for score, is_keep in zip(processed_pred_score_list, pred_str_filter) if is_keep]

            # A boolean np array indicates whether each prediction match the target after stemming before remove the one-word predictions
            # match_list = compute_match_result(true_str_list=trg_str_list, pred_str_list=processed_pred_str_list)

            topk_range = [5, 10]

            # only keep the first one-word prediction, remove all the remaining keyphrases that only has one word.
            filtered_pred_str_list, num_one_word_seqs = filter_one_word_sequences(processed_pred_str_list)
            print_out += "%d one-word sequences found, %d removed" % (num_one_word_seqs, num_one_word_seqs-1)

            # A boolean np array indicates whether each prediction match the target after stemming
            is_match = compute_match_result(trg_str_list=stemmed_trg_str_list, pred_str_list=stemmed_pred_str_list)

            num_filtered_predictions = len(filtered_pred_str_list)
            num_trg_str = len(trg_str_list)

            # Print out and store the recall, precision and F-1 score of every sample
            print_out += "Results:\n"
            for topk in topk_range:
                results = compute_classification_metrics_at_k(is_match, num_filtered_predictions, num_trg_str, topk=topk)
                for metric, result in results.items():
                    score_dict_all[metric].append(result)
                    print_out += "%s: %.3f\n" % (metric, result)

            total_predictions += num_filtered_predictions
            total_targets += num_trg_str
            if opt.verbose:
                logging.info(print_out)

    # Compute the micro averaged recall, precision and F-1 score
    #micro_avg_score_dict = {}
    for topk in topk_range:
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classificatioon_metrics(score_dict_all['num_matches@%d' % topk], total_predictions, total_targets)
        logging.info('micro_avg_precision@%d: %.3f' % (topk, micro_avg_precision_k))
        logging.info('micro_avg_recall@%d: %.3f' % (topk, micro_avg_recall_k))
        logging.info('micro_avg_f1_score@%d: %.3f' % (topk, micro_avg_f1_score_k))
        #micro_avg_score_dict['micro_avg_precision@%d' % topk]
        #micro_avg_score_dict['micro_avg_recall@%d' % topk]
        #micro_avg_score_dict['micro_avg_f1_score@%d' % topk]

    # Compute the macro averaged recall, precision and F-1 score
    for topk in topk_range:
        macro_avg_precision_k = np.array(score_dict_all['precision@%d']).mean()
        marco_avg_recall_k = np.array(score_dict_all['recall@%d']).mean()
        marco_avg_f1_score_k = float(2*macro_avg_precision_k*marco_avg_recall_k)/(macro_avg_precision_k+marco_avg_recall_k)
        logging.info('micro_avg_precision@%d: %.3f' % (topk, macro_avg_precision_k))
        logging.info('micro_avg_recall@%d: %.3f' % (topk, marco_avg_recall_k))
        logging.info('micro_avg_f1_score@%d: %.3f' % (topk, marco_avg_f1_score_k))

if __name__ == '__main__':
    src_str = ['this', 'is', 'a', 'short', 'paragraph', 'for', 'identifying', 'key', 'value', 'pairs', '.']
    keyphrase_str_list = [['short', 'paragraph'], ['short', 'paragraphs'], ['test', 'propose'], ['test', 'proposes']]
    is_present, not_duplicate, stemmed_keyphrase_str_list = check_present_and_duplicate_keyphrases(src_str, keyphrase_str_list)
    print(is_present)
    print(not_duplicate)
    print(stemmed_keyphrase_str_list)
