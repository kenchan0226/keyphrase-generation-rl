import numpy as np
from utils.string_helper import *
from evaluate_prediction import *
import torch


def sample_list_to_str_2dlist(sample_list, oov_lists, idx2word, vocab_size, eos_idx, delimiter_word):
    """Convert a list of sample dict to a 2d list of predicted keyphrases"""
    pred_str_2dlist = []  #  a 2dlist, len(pred_str_2d_list)=batch_size, len(pred_str_2d_list[0])=
    for sample, oov in zip(sample_list, oov_lists):
        word_list = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx)
        pred_str_list = split_concated_keyphrases(word_list, delimiter_word)
        pred_str_2dlist.append(pred_str_list)
    return pred_str_2dlist


def compute_reward(trg_str_2dlist, pred_str_2dlist, batch_size, reward_type='f1', topk=10, match_type="exact"):
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
        if reward_type == 0:  # f1
            # boolean np array to indicate which prediction matches the target
            is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                            pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
            precision_k, recall_k, f1_k, _, _ = compute_classification_metrics_at_k(is_match, num_unique_predictions, num_unique_targets, topk=topk)
            reward[idx] = f1_k
        elif reward_type == 1:  # recall
            # boolean np array to indicate which prediction matches the target
            is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                            pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
            precision_k, recall_k, f1_k, _, _ = compute_classification_metrics_at_k(is_match, num_unique_predictions,
                                                                                    num_unique_targets, topk=topk)
            reward[idx] = recall_k
        elif reward_type == 2:  # ndcg
            # boolean np array to indicate which prediction matches the target
            is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                            pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
            ndcg_k = ndcg_at_k(is_match, topk, method=1)
            reward[idx] = ndcg_k
        elif reward_type == 3:  # accuracy
            # boolean np array to indicate which prediction matches the target
            is_match = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                            pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=1)
            acc = sum(is_match)/is_match.shape[0]
            reward[idx] = acc
        elif reward_type == 4:  # alpha-ndcg
            # boolean np array to indicate which prediction matches the target
            is_match_2d = compute_match_result(trg_str_list=unique_stemmed_trg_str_list,
                                            pred_str_list=unique_stemmed_pred_str_list, type=match_type, dimension=2)
            # is_match_2d: [num_trg_str, num_pred_str]
            reward[idx] = alpha_ndcg_at_k(r_2d, topk, method=1, alpha=0.5)
    return reward


def compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, num_predictions, reward_shaping, reward_type, topk, match_type="exact"):
    phrase_reward = np.zeros((batch_size, num_predictions))
    if reward_shaping:
        for t in range(num_predictions):
            pred_str_2dlist_at_t = [pred_str_list[:t + 1] for pred_str_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_reward(trg_str_2dlist, pred_str_2dlist_at_t, batch_size, reward_type, topk, match_type)
    else:
        phrase_reward[:, num_predictions - 1] = compute_reward(trg_str_2dlist, pred_str_2dlist, batch_size, reward_type,
                                                               topk, match_type)
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
    print(alpha_ndcg_at_k(r_2d, 9))
    r_2d = r_2d[:, np.array([0, 4, 6, 1, 5, 2, 7, 8, 9])]
    print(alpha_ndcg_at_k(r_2d, 9))
    pass
