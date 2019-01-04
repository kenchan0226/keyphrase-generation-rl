import os
from tqdm import tqdm
import argparse


def filter_dups(saved_home, dups_info_home, context_file_path, keyword_file_path):
    """
    filter out the duplicates in the training data with the testing data according to the obtained duplication info file.
    :param saved_home: non-filtered data home
    :param dups_info_home: duplication information home
    :return: None
    """
    #orig_context_file = open(os.path.join(saved_home, 'data_for_corenlp', 'kp20k_training_context_for_corenlp.txt'),
    #                         encoding='utf-8')
    #context_lines = orig_context_file.readlines()
    #orig_allkeys_file = open(os.path.join(saved_home, 'data_for_corenlp', 'kp20k_training_keyword_for_corenlp.txt'),
    #                         encoding='utf-8')

    orig_context_file = open(context_file_path, encoding='utf-8')
    context_lines = orig_context_file.readlines()
    orig_allkeys_file = open(keyword_file_path, encoding='utf-8')
    allkeys_lines = orig_allkeys_file.readlines()
    assert len(context_lines) == len(allkeys_lines)

    context_file_name = os.path.split(context_file_path)[1]
    context_file_name = os.path.splitext(context_file_name)[0]
    filtered_context_file_name = "{}_filtered".format(context_file_name)
    keyword_file_name = os.path.split(keyword_file_path)[1]
    keyword_file_name = os.path.splitext(keyword_file_name)[0]
    filtered_keyword_file_name = "{}_filtered".format(keyword_file_name)

    # filter out the duplicates in the validation and the testing datasets and the kp20k training dataset itself
    dups_info_datasets = ['kp20k_training', 'kp20k_validation', 'kp20k_testing',
                          'inspec_testing', 'krapivin_testing',
                          'nus_testing', 'semeval_testing']
    total_filtered_idx_set = set()
    for dataset in dups_info_datasets:
        filtered_idx_set = set()
        dups_info_file = open(
            os.path.join(dups_info_home, '{}_context_nstpws_dups_w_kp20k_training.txt'.format(dataset)), encoding='utf-8')
        for line in dups_info_file:
            line = line.strip()
            # inspec_testing_48 kp20k_training_433051 jc_sc:0.7368; affine invariants of convex polygons | affine invariants of convex polygons
            dups, titles = line.split(';')
            src_dup, filtered_dup, _ = dups.split()
            src_idx = int(src_dup.strip().split('_')[-1])
            filtered_idx = int(filtered_dup.strip().split('_')[-1])
            if dataset != 'kp20k_training':
                filtered_idx_set.add(filtered_idx)
            else:
                if src_idx not in filtered_idx_set:
                    filtered_idx_set.add(filtered_idx)
        total_filtered_idx_set = total_filtered_idx_set.union(filtered_idx_set)
        print('Num of filtered kp20k training data: {}'.format(len(total_filtered_idx_set)))

    # also filter out the invalid data samples
    print('Finding the invalid data samples in the original kp20k training ...')
    for corpus_idx in tqdm(range(len(context_lines))):
        if context_lines[corpus_idx].strip().split() == [''] or allkeys_lines[corpus_idx].strip().split(' ; ') == ['']:
            total_filtered_idx_set.add(corpus_idx)
    print('Num of filtered kp20k training data: {}'.format(len(total_filtered_idx_set)))

    total_filtered_idxes = sorted(list(total_filtered_idx_set))
    for filter_idx in total_filtered_idxes:
        context_lines[filter_idx] = '\n'
        allkeys_lines[filter_idx] = '\n'

    filtered_context_file = open(os.path.join(saved_home, 'data_for_corenlp',
                                              '{}.txt'.format(filtered_context_file_name)),
                                 'w', encoding='utf-8')
    filtered_context_file.writelines(context_lines)

    filtered_allkeys_file = open(os.path.join(saved_home, 'data_for_corenlp',
                                              '{}.txt'.format(filtered_keyword_file_name)),
                                 'w', encoding='utf-8')
    filtered_allkeys_file.writelines(allkeys_lines)

    orig_context_file = open(os.path.join(saved_home, 'data_for_corenlp',
                                          'kp20k_training_filtered_for_corenlp_idxes.txt'),
                             'w', encoding='utf-8')
    orig_context_file.write(' '.join([str(idx) for idx in total_filtered_idxes]) + '\n')
    orig_context_file.write(str(len(total_filtered_idxes)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='integrated_data_preprocess')
    parser.add_argument('-saved_home', type=str, default='process_json/integrated_processed_data')
    parser.add_argument('-context_file_path', type=str,
                        default='process_json/integrated_processed_data/data_for_corenlp/kp20k_training_context_for_corenlp_sorted.txt')
    parser.add_argument('-keyword_file_path', type=str,
                        default='process_json/integrated_processed_data/data_for_corenlp/kp20k_training_keyword_for_corenlp_sorted.txt')
    parser.add_argument('-dups_info_home', type=str,
                        default='process_json/duplicates_w_kp20k_training')
    opts = parser.parse_args()
    filter_dups(saved_home=opts.saved_home, dups_info_home=opts.dups_info_home, context_file_path=opts.context_file_path, keyword_file_path=opts.keyword_file_path)

