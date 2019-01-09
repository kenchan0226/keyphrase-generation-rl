import argparse
from integrated_data_preprocess import check_present_idx
import numpy as np
from utils import string_helper
import os

present_absent_segmenter = '<peos>'


def find_present_idx_for_variation_list(src_tokens, variations_str_list):
    src_len = len(src_tokens)
    src_tokens_stemmed = string_helper.stem_word_list(src_tokens)
    present_flag = False
    min_present_idx = 10 * src_len
    for variation_str in variations_str_list:
        variation_tokens = variation_str.split(' ')
        variation_tokens_stemmed = string_helper.stem_word_list(variation_tokens)
        present_idx, is_present = check_present_idx(src_tokens_stemmed, variation_tokens_stemmed)
        if present_idx < min_present_idx:
            min_present_idx = present_idx
        if is_present:
            present_flag = True
    return min_present_idx, present_flag


def sort_keyphrases_with_variations(src_tokens, target_str_list):
    src_len = len(src_tokens)
    num_trgs = len(target_str_list)
    present_indices_array = np.ones(num_trgs) * (src_len + 1)
    num_present_keyphrases = 0
    for i, target_str in enumerate(target_str_list):
        variations_str_list = target_str.split('|')
        present_indices_array[i], is_present = find_present_idx_for_variation_list(src_tokens, variations_str_list)
        if is_present:
            num_present_keyphrases += 1
    sorted_keyphrase_indices = np.argsort(present_indices_array)
    sorted_keyphrase_list = [target_str_list[idx] for idx in sorted_keyphrase_indices]
    sorted_keyphrase_list.insert(num_present_keyphrases, present_absent_segmenter)
    return sorted_keyphrase_list


def main(src_file_path, trg_file_path, saved_home):
    trg_file_name = os.path.split(trg_file_path)[1]
    trg_file_name = os.path.splitext(trg_file_name)[0]
    sorted_trg_file_name = "{}_sorted_separated.txt".format(trg_file_name)
    sorted_trg_file = open(os.path.join(saved_home, 'data_for_corenlp', sorted_trg_file_name), 'w',  encoding='utf-8')

    for src_line, trg_line in zip(open(src_file_path, 'r'), open(trg_file_path, 'r')):
        src_tokens = src_line.strip().split(' ')
        target_str_list = trg_line.strip().split(';')
        sorted_keyphrase_list = sort_keyphrases_with_variations(src_tokens, target_str_list)
        sorted_trg_file.write(';'.join(sorted_keyphrase_list) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sort_testing_data')
    parser.add_argument('-saved_home', type=str)
    parser.add_argument('-context_file_path', type=str)
    parser.add_argument('-keyword_file_path', type=str)
    opts = parser.parse_args()
    main(opts.context_file_path, opts.keyword_file_path, opts.saved_home)
