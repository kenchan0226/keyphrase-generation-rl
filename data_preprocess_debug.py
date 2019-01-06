from integrated_data_preprocess import *


def sort_keyphrases_by_their_order_of_occurence_debug():
    keyphrase_list = ["svm", "we", "support vector machine", "in this", "pca", "abcdefgh", ""]
    source = "In this paper, we support propose a support vector machine to classify."
    src_tokens = get_tokens(source.lower(), True, False)
    keyphrase_token_2dlist = [get_tokens(keyphrase.lower(), True, False) for keyphrase in keyphrase_list]
    print(sort_keyphrases_by_their_order_of_occurence(keyphrase_list, src_tokens, keyphrase_token_2dlist, separate_present_absent=True))
    return


def check_present_idx_debug():
    keyphrase_list = ["svm", "support", "support vector machine", ".", "in this", "pca", "suppor", ""]
    # keyphrase_list = ["svm", "we", "support vector machine", "in this", "pca", "abcdefgh", ""]
    source = "In this paper, we support propose a support vector machine to classify."
    keyphrase_list_tokenized = [get_tokens(keyphrase.lower(), True, False) for keyphrase in keyphrase_list]
    source_tokens = get_tokens(source.lower(), True, False)
    print(source_tokens)
    src_len = len(source_tokens)
    print(src_len)
    print(batch_check_present_idx(source_tokens, keyphrase_list_tokenized))
    return


def find_variations_from_wiki_debug():
    keyphrase_list = ["svm", "support vector machine", "principal component analysis", "abcdefg", "pca"]
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True, False)
    for keyphrase in keyphrase_list:
        print(find_variations_from_wiki(keyphrase, source_tokens, True))
    return


def find_variations_debug():
    keyphrase_list = ["svm", "support vector machine", "principal component analysis", "abcdefg", "pca", "principal component analysis (pca)", "apple (a b c d e f g h)"]
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True, False)
    for keyphrase in keyphrase_list:
        print(find_variations(keyphrase, source_tokens, fine_grad=True, limit_num=True))
    return


def process_keyphrase_debug_with_variation():
    keyphrase_list = ["svm", "support vector machine", "principal component analysis", "abcdefg", "pca", "principal component analysis (pca)", "apple (a b c d e f g h)", ""]
    keyphrase_str = ';'.join(keyphrase_list)
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True, False)
    variate_keyphrase_list = process_keyphrase(keyphrase_str, source_tokens, variations=True, fine_grad=True, limit_num=True)
    print(variate_keyphrase_list)
    for keyphrase in variate_keyphrase_list:
        print(keyphrase)
    #print(process_keyphrase(keyphrase_str, source_tokens, variations=True, fine_grad=True, limit_num=True))
    """
    variate_keyphrase_str = process_keyphrase(keyphrase_str, source_tokens, variations=True, fine_grad=True, limit_num=True)
    variate_keyphrase_list = variate_keyphrase_str.split(';')
    for keyphrase in variate_keyphrase_list:
        print(keyphrase)
    """
    return


def process_keyphrase_debug_sort_keyphrases():
    keyphrase_list = ["svm", "we", "support vector machine", "in this", "pca", "abcdefgh", ""]
    keyphrase_str = ';'.join(keyphrase_list)
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True, False)
    variate_keyphrase_list = process_keyphrase(keyphrase_str, source_tokens, variations=False, fine_grad=True, limit_num=True, sort_keyphrases=True)
    print(variate_keyphrase_list)
    for keyphrase in variate_keyphrase_list:
        print(keyphrase)
    #print(process_keyphrase(keyphrase_str, source_tokens, variations=True, fine_grad=True, limit_num=True))
    """
    variate_keyphrase_str = process_keyphrase(keyphrase_str, source_tokens, variations=True, fine_grad=True, limit_num=True)
    variate_keyphrase_list = variate_keyphrase_str.split(';')
    for keyphrase in variate_keyphrase_list:
        print(keyphrase)
    """
    return


if __name__ == '__main__':
    #process_keyphrase_debug_with_variation()
    check_present_idx_debug()
    #sort_keyphrases_by_their_order_of_occurence_debug()
    #process_keyphrase_debug_sort_keyphrases()

