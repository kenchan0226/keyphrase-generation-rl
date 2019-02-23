import pickle
import numpy as np
from scipy import stats
import argparse


def main(score_dict_a, score_dict_b, k_list, tag_list):
    for tag in tag_list:
        for k in k_list:
            f1_np_array_a = np.array(score_dict_a['f1_score@{}_{}'.format(k, tag)])
            f1_np_array_b = np.array(score_dict_b['f1_score@{}_{}'.format(k, tag)])
            t_stat, p_value = stats.ttest_rel(f1_np_array_a, f1_np_array_b)
            print("tag: {}, topk: {}, p-value: {}".format(tag, k, p_value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t_stat')
    parser.add_argument('-score_dict_path_a', type=str, default='')
    parser.add_argument('-score_dict_path_b', type=str, default='')
    parser.add_argument('-k_list', nargs='+', default=['5', 'M'], type=str, help='only allow integer or M')
    opt = parser.parse_args()
    pickle_in_a = open(opt.score_dict_path_a, "rb")
    pickle_in_b = open(opt.score_dict_path_b, "rb")
    score_dict_a = pickle.load(pickle_in_a)
    score_dict_b = pickle.load(pickle_in_b)
    tag_list = ['present', 'absent']
    main(score_dict_a, score_dict_b, opt.k_list, tag_list)
