#from nltk.stem.porter import *
import torch
#from utils import Progbar
#from pykp.metric.bleu import bleu
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics, RewardStatistics
import time
from utils.time_log import time_since
#from nltk.stem.porter import *
import pykp
import logging
import numpy as np
from collections import defaultdict
import os
import sys
from utils.string_helper import *
from pykp.reward import sample_list_to_str_2dlist, compute_batch_reward

#stemmer = PorterStemmer()

def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            if not opt.one2many:  # load one2one dataset
                src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, title, title_oov, title_lens, title_mask = batch
            else:  # load one2many dataset
                src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
                num_trgs = [len(trg_str_list) for trg_str_list in
                            trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

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
                # title_oov = title_oov.to(opt.device)

            start_time = time.time()
            if not opt.one2many:
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _, _ = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, title=title, title_lens=title_lens, title_mask=title_mask)
            else:
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _, _ = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, num_trgs, title=title, title_lens=title_lens, title_mask=title_mask)
            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, coverage_loss=False)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, coverage_loss=False)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total, loss_compute_time=loss_compute_time_total)
    return eval_loss_stat

def evaluate_reward(data_loader, generator, opt):
    """Return the avg. reward in the validation dataset"""
    generator.model.eval()
    final_reward_sum = 0.0
    n_batch = 0
    sample_time_total = 0.0
    topk = opt.topk
    reward_type = opt.reward_type
    #reward_type = 7
    match_type = opt.match_type
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            # load one2many dataset
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
            num_trgs = [len(trg_str_list) for trg_str_list in
                        trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            #trg = trg.to(opt.device)
            #trg_mask = trg_mask.to(opt.device)
            #trg_oov = trg_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)

            start_time = time.time()
            # sample a sequence
            # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
            sample_list, log_selected_token_dist, output_mask, pred_idx_mask, _, _, _ = generator.sample(
                src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=True, one2many=one2many,
                one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=0, title=title, title_lens=title_lens, title_mask=title_mask)
            #pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word)
            pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                        delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                        src_str_list)
            #print(pred_str_2dlist)
            sample_time = time_since(start_time)
            sample_time_total += sample_time

            final_reward = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor=0.0)  # np.array, [batch_size]

            final_reward_sum += final_reward.sum(0)

    eval_reward_stat = RewardStatistics(final_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)

    return eval_reward_stat

"""
def prediction_by_sampling(generator, data_loader, opt, delimiter_word):
    # file for storing the predicted keyphrases
    if opt.pred_file_prefix == "":
        pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    else:
        pred_output_file = open(os.path.join(opt.pred_path, "%s_predictions.txt" % opt.pred_file_prefix), "w")
    # debug
    interval = 1000
    generator.model.eval()

    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (batch_i+1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()

            # load one2many dataset
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, original_idx_list = batch
            num_trgs = [len(trg_str_list) for trg_str_list in
                        trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            # trg = trg.to(opt.device)
            # trg_mask = trg_mask.to(opt.device)
            # trg_oov = trg_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)

            eos_idx = opt.word2idx[pykp.io.EOS_WORD]
            batch_size = src.size(0)

            one2many = opt.one2many
            one2many_mode = opt.one2many_mode
            '''
            if one2many and one2many_mode == 2:
                num_predictions = opt.n_best
            else:
                num_predictions = 1
            '''
            num_predictions = opt.max_eos_per_output_seq
            start_time = time.time()
            # sample a sequence
            # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
            sample_list, log_selected_token_dist, output_mask, pred_idx_mask, _, _, _ = generator.sample(
                src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
                one2many_mode=one2many_mode, num_predictions=num_predictions)
            pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                        delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk, src_str_list)
            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, pred_str_2dlist),
                               key=lambda p: p[0])
            original_idx_list, pred_str_2dlist = zip(*seq_pairs)

            # output the predicted keyphrases to a file
            pred_print_out = ''
            for pred_str_list in pred_str_2dlist:
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
            pred_output_file.write(pred_print_out)

    pred_output_file.close()
    print("done!")
    return pred_str_2dlist
"""

'''
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


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            #sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def evaluate_beam_search(generator, one2many_data_loader, opt, delimiter_word='<sep>'):
    #score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    # file for storing the predicted keyphrases
    if opt.pred_file_prefix == "":
        pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    else:
        pred_output_file = open(os.path.join(opt.pred_path, "%s_predictions.txt" % opt.pred_file_prefix), "w")
    # debug
    interval = 1000

    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (batch_i+1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, _, _, _, _, original_idx_list, title, title_oov, title_lens, title_mask = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)


            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx, opt.max_eos_per_output_seq, title=title, title_lens=title_lens, title_mask=title_mask)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists, opt.word2idx[pykp.io.EOS_WORD], opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk, src_str_list)
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred['sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len], does not include the final <EOS>
                pred_score_list = pred['scores']
                pred_attn_list = pred['attention']  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]

                if opt.one2many:
                    all_keyphrase_list = []  # a list of word list contains all the keyphrases in the top max_n sequences decoded by beam search
                    for word_list in pred_str_list:
                        all_keyphrase_list += split_word_list_by_delimiter(word_list, delimiter_word, opt.separate_present_absent, pykp.io.PEOS_WORD)
                        #not_duplicate_mask = check_duplicate_keyphrases(all_keyphrase_list)
                    #pred_str_list = [word_list for word_list, is_keep in zip(all_keyphrase_list, not_duplicate_mask) if is_keep]
                    pred_str_list = all_keyphrase_list

                # output the predicted keyphrases to a file
                pred_print_out = ''
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
                pred_output_file.write(pred_print_out)

    pred_output_file.close()
    print("done!")


'''
def evaluate_beam_search_one2many(generator, one2many_data_loader, opt):
    score_dict_all = defaultdict(
        list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    # file for storing the predicted keyphrases
    pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    # debug
    interval = 1000

    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (
                batch_i + 1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, _, _, _, _, original_idx_list = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                      opt.word2idx[pykp.io.EOS_WORD])
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred[
                    'sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len]
                pred_score_list = pred['scores']
                pred_attn_list = pred['attention']

                # output the predicted keyphrases to a file
                pred_print_out = ''
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
                pred_output_file.write(pred_print_out)

    pred_output_file.close()
    print("done!")
'''

'''
def evaluate_beam_search_backup(generator, one2many_data_loader, opt, save_path=None):
    score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    # file for storing the predicted keyphrases
    pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    # debug
    interval = 1000

    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (batch_i+1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, _, _, _, _, original_idx_list = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists, opt.word2idx[pykp.io.EOS_WORD])
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists), key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred['sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len]
                pred_score_list = pred['scores']
                pred_attn_list = pred['attention']

                # output the predicted keyphrases to a file
                pred_print_out = ''
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
                pred_output_file.write(pred_print_out)

                verbose_print_out = '[Source][%d]: %s \n' % (len(src_str), ' '.join(src_str))

                # stem the src str, trg str list and pred_str_list
                stemmed_src_str = stem_word_list(src_str)
                stemmed_trg_str_list = stem_str_list(trg_str_list)
                stemmed_pred_str_list = stem_str_list(pred_str_list)

                # is_present: boolean np array indicate whether a predicted keyphrase is present in src
                # not_duplicate: boolean np array indicate
                trg_str_is_present, trg_str_not_duplicate = check_present_and_duplicate_keyphrases(stemmed_src_str, stemmed_trg_str_list)

                #verbose_print_out += '[GROUND-TRUTH] #(present)/#(all targets)=%d/%d\n' % (sum(trg_str_is_present), len(trg_str_list))
                #verbose_print_out += '\n'.join(
                #    ['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in
                #     zip(trg_str_list, trg_str_is_present)])
                #verbose_print_out += '\noov_list:   \n\t\t%s \n' % str(oov)

                # a pred_seq is invalid if len(processed_seq) == 0 or keep_flag and any word in processed_seq is UNK or it contains '.' or ','
                pred_str_is_valid = check_valid_keyphrases(pred_str_list)
                #pred_str_is_valid, processed_pred_seq_list, processed_pred_str_list, processed_pred_score_list = process_predseqs(pred_seq_list, oov, opt.idx2word, opt)

                # a list of boolean indicates which predicted keyphrases present in src, for the duplicated keyphrases after stemming, only consider the first one
                pred_str_is_present, pred_str_not_duplicate = check_present_and_duplicate_keyphrases(stemmed_src_str, stemmed_pred_str_list)

                # print out all the predicted keyphrases
                verbose_print_out += '[All PREDICTIONs] #(valid)=%d, #(present)=%d, #(unique)=%d, #(all)=%d\n' % (
                    sum(pred_str_is_valid), sum(pred_str_is_present), sum(pred_str_not_duplicate), len(pred_str_list))
                for word_list, is_present in zip(pred_str_list, pred_str_is_present):
                    if is_present:
                        verbose_print_out += '\t\t[%s]' % ' '.join(word_list)
                    else:
                        verbose_print_out += '\t\t%s' % ' '.join(word_list)
                verbose_print_out += '\n'

                # Only keep the first one-word prediction, remove all the remaining keyphrases that only has one word.
                extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_str_list)
                verbose_print_out += "%d one-word sequences found, %d removed\n" % (num_one_word_seqs, num_one_word_seqs - 1)

                tmp_trg_str_filter = trg_str_not_duplicate
                tmp_pred_str_filter = pred_str_not_duplicate * pred_str_is_valid * extra_one_word_seqs_mask

                # Filter for present keyphrase prediction
                trg_str_filter_present = tmp_trg_str_filter * trg_str_is_present
                pred_str_filter_present = tmp_pred_str_filter * pred_str_is_present

                # Filter for absent keyphrase prediction
                trg_str_filter_absent = tmp_trg_str_filter * np.invert(trg_str_is_present)
                pred_str_filter_absent = tmp_pred_str_filter * np.invert(pred_str_is_present)

                # A list to store all the predicted keyphrases after filtering for both present and absent keyphrases
                filtered_pred_str_list_present_and_absent = []

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
                    filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in zip(stemmed_trg_str_list, trg_str_filter)
                                                     if
                                                     is_keep]

                    #processed_pred_seq_list = [seq for seq, is_keep in zip(processed_pred_seq_list, pred_str_filter) if is_keep]
                    filtered_pred_str_list = [word_list for word_list, is_keep in zip(pred_str_list, pred_str_filter) if
                                               is_keep]
                    filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in zip(stemmed_pred_str_list, pred_str_filter) if
                                               is_keep]
                    filtered_pred_score_list = [score for score, is_keep in zip(pred_score_list, pred_str_filter) if is_keep]
                    filtered_pred_attn_list = [attn for attn, is_keep in zip(pred_attn_list, pred_str_filter) if is_keep]

                    topk_range = [5, 10]

                    filtered_pred_str_list_present_and_absent += filtered_pred_str_list

                    # A boolean np array indicates whether each prediction match the target after stemming
                    is_match = compute_match_result(trg_str_list=filtered_stemmed_trg_str_list, pred_str_list=filtered_stemmed_pred_str_list)

                    num_filtered_predictions = len(filtered_pred_str_list)
                    num_filtered_targets = len(filtered_trg_str_list)

                    if (opt.attn_debug):
                        # TODO: for each prediction, print out the attention weight on each src for case studies
                        pass

                    # Print out and store the recall, precision and F-1 score of every sample
                    verbose_print_out += "Results (%s):\n" % present_tag
                    for topk in topk_range:
                        precision_k, recall_k, f1_k, num_matches_k, num_predictions_k = \
                            compute_classification_metrics_at_k(is_match, num_filtered_predictions, num_filtered_targets, topk=topk)
                        results = prepare_classification_result_dict(precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, num_filtered_targets, topk, is_present)
                        for metric, result in results.items():
                            score_dict_all[metric].append(result)
                            verbose_print_out += "%s: %.3f\n" % (metric, result)

                        #print("Result of %s@%d" % (present_tag, topk))
                        #print(results)

                #print("Result dict all:")
                #print(score_dict_all)

                if opt.verbose:
                    logging.info(verbose_print_out)
                    #print(verbose_print_out)
                    #sys.stdout.flush()

    pred_output_file.close()

    # Compute the micro averaged recall, precision and F-1 score
    #micro_avg_score_dict = {}

    start_time = time.time()
    print("Start writing log.")

    for is_present in [True, False]:
        present_tag = 'present' if is_present else 'absent'
        logging.info('Final Results (%s):' % present_tag)
        for topk in topk_range:
            total_predictions = sum(score_dict_all['num_predictions@%d_%s' % (topk, present_tag)])
            total_targets = sum(score_dict_all['num_targets@%d_%s' % (topk, present_tag)])
            # Compute the micro averaged recall, precision and F-1 score
            micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classificatioon_metrics(sum(score_dict_all['num_matches@%d_%s' % (topk, present_tag)]), total_predictions, total_targets)
            logging.info('micro_avg_precision@%d_%s:%.3f' % (topk, present_tag, micro_avg_precision_k))
            logging.info('micro_avg_recall@%d_%s:%.3f' % (topk, present_tag, micro_avg_recall_k))
            logging.info('micro_avg_f1_score@%d_%s:%.3f' % (topk, present_tag, micro_avg_f1_score_k))
            # Compute the macro averaged recall, precision and F-1 score
            macro_avg_precision_k = sum(score_dict_all['precision@%d_%s' % (topk, present_tag)])/len(score_dict_all['precision@%d_%s' % (topk, present_tag) ])
            marco_avg_recall_k = sum(score_dict_all['recall@%d_%s' % (topk, present_tag)])/len(score_dict_all['recall@%d_%s' % (topk, present_tag) ])
            marco_avg_f1_score_k = float(2*macro_avg_precision_k*marco_avg_recall_k)/(macro_avg_precision_k+marco_avg_recall_k)
            logging.info('macro_avg_precision@%d_%s: %.3f' % (topk, present_tag, macro_avg_precision_k))
            logging.info('macro_avg_recall@%d_%s: %.3f' % (topk, present_tag, marco_avg_recall_k))
            logging.info('macro_avg_f1_score@%d_%s: %.3f' % (topk, present_tag, marco_avg_f1_score_k))

    print('Time for writing log: %.1f' % time_since(start_time))
    sys.stdout.flush()
'''

if __name__ == '__main__':
    pass
    '''
    src_str = ['this', 'is', 'a', 'short', 'paragraph', 'for', 'identifying', 'key', 'value', 'pairs', '.']
    keyphrase_str_list = [['short', 'paragraph'], ['short', 'paragraphs'], ['test', 'propose'], ['test', 'proposes']]
    is_present, not_duplicate, stemmed_keyphrase_str_list = check_present_and_duplicate_keyphrases(src_str, keyphrase_str_list)
    print(is_present)
    print(not_duplicate)
    print(stemmed_keyphrase_str_list)
    '''
    '''
    src_str_list = [['this', 'is', 'a', 'short', 'paragraph', 'for', 'identifying', 'key', 'value', 'pairs', '.'],
                    ['thanks', 'god', 'this', 'is', 'friday','.']]
    trg_str_2dlist = [[['short', 'paragraph'], ['short', 'paragraphs'], ['test', 'propose'], ['test', 'proposes'],
               ['demo purpose']],
               [['happy', 'friday'], ['break'], ['hang', 'out']]]
    pred_str_2dlist = [[['short', 'paragraph'], ['short', 'paragraphs'], ['test', 'propose'], ['test', 'proposes'], ['is'],
                ['a'], ['apple'], ['singing', 'contest'], ['orange', '.']],
                [['happy', 'friday'], ['break'], ['prison'], ['hand', 'shake']]]

    for src_str, trg_str_list, pred_str_list in zip(src_str_list, trg_str_2dlist, pred_str_2dlist):
        stemmed_src = stem_word_list(src_str)
        print(stemmed_src)
        stemmed_trg_str_list = stem_str_list(trg_str_list)
        print(stemmed_trg_str_list)
        stemmed_pred_str_list = stem_str_list(pred_str_list)
        print(stemmed_pred_str_list)
        trg_is_present, trg_not_duplicate = check_present_and_duplicate_keyphrases(stemmed_src, stemmed_trg_str_list)
        print(trg_is_present)
        print(trg_not_duplicate)
        pred_is_present, pred_not_duplicate = check_present_and_duplicate_keyphrases(stemmed_src, stemmed_pred_str_list)
        print(pred_is_present)
        print(pred_not_duplicate)
    '''