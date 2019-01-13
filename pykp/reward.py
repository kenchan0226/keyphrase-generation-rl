import numpy as np
from utils.string_helper import *
from evaluate_prediction import *
import torch


def sample_list_to_str_2dlist(sample_list, oov_lists, idx2word, vocab_size, eos_idx, delimiter_word, unk_idx=None, replace_unk=False, src_str_list=None, separate_present_absent=False, present_absent_delimiter_word=None):
    """Convert a list of sample dict to a 2d list of predicted keyphrases"""
    pred_str_2dlist = []  #  a 2dlist, len(pred_str_2d_list)=batch_size, len(pred_str_2d_list[0])=
    for sample, oov, src_word_list in zip(sample_list, oov_lists, src_str_list):
        # sample['prediction']: list of 0-dim tensor, len=trg_len
        # sample['attention']: tensor with size [trg_len, src_len]
        word_list = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, sample['attention'])
        pred_str_list = split_word_list_by_delimiter(word_list, delimiter_word, separate_present_absent, present_absent_delimiter_word)
        #pred_str_list = split_concated_keyphrases(word_list, delimiter_word)
        pred_str_2dlist.append(pred_str_list)
    return pred_str_2dlist


def compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type='f1', topk=10, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    assert len(trg_str_2dlist) == batch_size
    assert len(pred_str_2dlist) == batch_size
    reward = np.zeros(batch_size)

    if regularization_type == 2:
        if entropy is None:
            raise ValueError('Entropy should not be none when regularization type is 2')
        assert reward.shape[0] == entropy.shape[0]

    for idx, (trg_str_list, pred_str_list) in enumerate(zip(trg_str_2dlist, pred_str_2dlist)):
        # trg_str_list, list of word list, len = number of target keyphrases, trg_str_list[i] = word list of i-th target keyphrase
        # pred_str_list, list of word list, len = number of predicted keyphrases
        if entropy is None:
            entropy_idx = None
        else:
            entropy_idx = entropy[idx]
        reward[idx] = compute_reward(pred_str_list, trg_str_list, reward_type, topk, match_type, regularization_factor, regularization_type, entropy_idx)
    return reward


def compute_reward(pred_str_list, trg_str_list, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    num_predictions = len(pred_str_list)
    # perform stemming
    stemmed_trg_str_list = stem_str_list(trg_str_list)
    stemmed_pred_str_list = stem_str_list(pred_str_list)

    trg_str_unique_filter = check_duplicate_keyphrases(
        stemmed_trg_str_list)  # a boolean nparray, true if not duplicated
    pred_str_unique_filter = check_duplicate_keyphrases(stemmed_pred_str_list)

    unique_stemmed_trg_str_list = [word_list for word_list, is_keep in zip(stemmed_trg_str_list, trg_str_unique_filter)
                                   if
                                   is_keep]
    unique_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                    zip(stemmed_pred_str_list, pred_str_unique_filter) if
                                    is_keep]
    num_unique_targets = len(unique_stemmed_trg_str_list)
    num_unique_predictions = len(unique_stemmed_pred_str_list)

    # replace all duplicate keyphrases by a <PAD> token, i.e., treat it as a incorrect keyphrase
    penalized_stemmed_pred_str_list = []
    for pred_word_list, is_unique in zip(stemmed_pred_str_list, pred_str_unique_filter):
        if is_unique:
            penalized_stemmed_pred_str_list.append(pred_word_list)
        else:
            penalized_stemmed_pred_str_list.append(['<pad>'])
    # ============

    if regularization_type == 1:
        """
        if num_predictions > 0:
            duplicate_predictions_fraction = 1 - num_unique_predictions/num_predictions
        else:
            duplicate_predictions_fraction = 1.0
        regularization = -duplicate_predictions_fraction
        """
        if num_predictions > 0:
            unique_prediction_fraction = num_unique_predictions / num_predictions
        else:
            unique_prediction_fraction = 0
        regularization = unique_prediction_fraction
    elif regularization_type == 2:
        regularization = entropy
    else:
        regularization = 0.0

    # regularization = regularization_factor * regularization

    if reward_type == 0:  # f1
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
        precision_k, recall_k, f1_k, _, _ = compute_classification_metrics_at_k(is_match, num_unique_predictions,
                                                                                num_unique_targets, topk=topk)
        tmp_reward = f1_k
    elif reward_type == 1:  # recall
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
        precision_k, recall_k, f1_k, _, _ = compute_classification_metrics_at_k(is_match, num_unique_predictions,
                                                                                num_unique_targets, topk=topk)
        tmp_reward = recall_k
    elif reward_type == 2:  # ndcg
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
        ndcg_k = ndcg_at_k(is_match, topk, method=1)
        tmp_reward = ndcg_k
    elif reward_type == 3:  # accuracy
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
        acc = sum(is_match) / is_match.shape[0]
        tmp_reward = acc
    elif reward_type == 4:  # alpha-ndcg
        # boolean np array to indicate which prediction matches the target
        is_match_2d = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                           pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=2)
        # is_match_2d: [num_trg_str, num_pred_str]
        tmp_reward = alpha_ndcg_at_k(is_match_2d, topk, method=1, alpha=0.5)
    elif reward_type == 5:  # alpha-dcg
        # boolean np array to indicate which prediction matches the target
        is_match_2d = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                           pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=2)
        # is_match_2d: [num_trg_str, num_pred_str]
        tmp_reward = alpha_dcg_at_k(is_match_2d, topk, method=1, alpha=0.5)
    elif reward_type == 6:  # average precision (AP)
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
        tmp_reward = average_precision_at_k(is_match, topk, num_unique_predictions, num_unique_targets)
    elif reward_type == 7:  # f1 while treating all duplication as incorrect guess
        # boolean np array to indicate which prediction matches the target
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=penalized_stemmed_pred_str_list, type=match_type, dimension=1)
        precision_k, recall_k, f1_k, _, _ = compute_classification_metrics_at_k(is_match, num_predictions,
                                                                                num_unique_targets, topk=topk)
        tmp_reward = f1_k
    elif reward_type == 8:  # AP while treating all duplication as incorrect guess
        is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                        pred_str_list=penalized_stemmed_pred_str_list, type=match_type, dimension=1)
        tmp_reward = average_precision_at_k(is_match, topk, num_predictions, num_unique_targets)

    # Add the regularization term to the reward only if regularization type != 0
    if regularization_type == 0 or regularization_factor == 0:
        reward = tmp_reward
    else:
        reward = (1 - regularization_factor) * tmp_reward + regularization_factor * regularization
        # reward[idx] += regularization
    return reward


def compute_present_absent_reward(pred_str_2dlist, trg_str_2dlist, reward_type='f1', topk=10, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    batch_size = len(pred_str_2dlist)
    present_absent_reward = np.zeros((batch_size, 2))

    for batch_idx, (pred_str_list, trg_str_list) in enumerate(zip(pred_str_2dlist, trg_str_2dlist)):
        pred_peos_position = 0
        trg_peos_position = 0
        pred_peos_counter = 0

        for pred_phrase_idx, pred_word_list in enumerate(pred_str_list):
            if pred_word_list[0] == pykp.io.PEOS_WORD:
                if pred_peos_counter == 0:  # only consider the first peos
                    pred_peos_position = pred_phrase_idx
                pred_peos_counter += 1

        present_pred_str_list = pred_str_list[0:pred_peos_position]
        absent_pred_str_list = pred_str_list[pred_peos_position+1:]

        for trg_phrase_idx, trg_word_list in enumerate(trg_str_list):
            if trg_word_list[0] == pykp.io.PEOS_WORD:
                trg_peos_position = trg_phrase_idx
        present_trg_str_list = trg_str_list[0:trg_peos_position]
        absent_trg_str_list = trg_str_list[trg_peos_position+1:]

        present_reward = compute_reward(present_pred_str_list, present_trg_str_list, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
        absent_reward = compute_reward(absent_pred_str_list, absent_trg_str_list, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)

        present_absent_reward[batch_idx, 0] = present_reward
        present_absent_reward[batch_idx, 1] = absent_reward

        # insert the reward of present prediction at the location of peos in the prediction str
        #stepwise_reward[batch_idx, location_of_peos_for_each_batch[batch_idx]] = present_reward

    return present_absent_reward


def present_absent_reward_to_stepwise_reward(present_absent_reward, max_pred_seq_len, location_of_peos_for_each_batch, location_of_eos_for_each_batch):
    batch_size = present_absent_reward.shape[0]
    stepwise_reward = np.zeros((batch_size, max_pred_seq_len))
    for batch_i in range(batch_size):
        # insert present reward to the location of <peos>
        stepwise_reward[batch_i, location_of_peos_for_each_batch[batch_i]] = present_absent_reward[batch_i, 0]
        # insert absent reward to the location of <eos>
        stepwise_reward[batch_i, location_of_eos_for_each_batch[batch_i]] = present_absent_reward[batch_i, 1]

    return stepwise_reward


def compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, max_num_phrases, reward_shaping, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    phrase_reward = np.zeros((batch_size, max_num_phrases))
    if reward_shaping:
        for t in range(max_num_phrases):
            pred_str_2dlist_at_t = [pred_str_list[:t + 1] for pred_str_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_batch_reward(pred_str_2dlist_at_t, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
    else:
        phrase_reward[:, -1] = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type,
                                                               topk, match_type, regularization_factor, regularization_type, entropy)
    return phrase_reward


def compute_phrase_reward_backup(pred_str_2dlist, trg_str_2dlist, batch_size, num_predictions, reward_shaping, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    phrase_reward = np.zeros((batch_size, num_predictions))
    if reward_shaping:
        for t in range(num_predictions):
            pred_str_2dlist_at_t = [pred_str_list[:t + 1] for pred_str_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_batch_reward(pred_str_2dlist_at_t, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
    else:
        phrase_reward[:, num_predictions - 1] = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type,
                                                               topk, match_type, regularization_factor, regularization_type, entropy)
    return phrase_reward


def shape_reward(reward_np_array):
    batch_size, seq_len = reward_np_array.shape
    left_padding = np.zeros((batch_size, 1))
    left_padded_reward = np.concatenate([left_padding, reward_np_array], axis=1)
    return np.diff(left_padded_reward, n=1, axis=1)


def phrase_reward_to_stepwise_reward(phrase_reward, eos_idx_mask):
    batch_size, seq_len = eos_idx_mask.size()
    stepwise_reward = np.zeros((batch_size, seq_len))
    for i in range(batch_size):
        pred_cnt = 0
        for j in range(seq_len):
            if eos_idx_mask[i, j].item() == 1:
                stepwise_reward[i, j] = phrase_reward[i, pred_cnt]
                pred_cnt += 1
            #elif j == seq_len:
            #    pass
    return stepwise_reward


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


if __name__ == "__main__":
    #reward = np.array([[1,3,5,6],[2,3,5,9]])
    #print(shape_reward(reward))

    #pred_str_list = [['multi', 'agent', 'system'], ['agent', 'warning'], ['multi', 'agent'], ['agent'], ['agent', 'system'], ['multi', 'system'], ['what', 'is']]
    #trg_str_list = [['multi', 'agent', 'system'], ['multi'], ['what', 'is']]
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='exact'))
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='sub'))
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='exact', dimension=2))
    #print(compute_match_result_new(trg_str_list, pred_str_list, type='sub', dimension=2))

    #r = np.array([2, 1, 2, 0])
    #print(ndcg_at_k(r, 4, method=1))  # 0.96519546960144276

    r_2d = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0, 0, 1, 1, 0, 1, 0, 0]])
    k_list = [1,2,3]
    print(alpha_ndcg_at_ks(r_2d, k_list))
    r_2d = r_2d[:, np.array([0, 4, 6, 1, 5, 2, 7, 8, 9])]
    print(alpha_ndcg_at_ks(r_2d, k_list))

    '''
    r = np.array([0,1,1,0,1,0])
    k_list = [4, 6]
    print(average_precision_at_ks(r, k_list, num_trgs=5, num_predictions=6))
    '''
    pass
