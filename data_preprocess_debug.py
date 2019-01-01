from integrated_data_preprocess import find_variations_from_wiki, get_tokens, find_variations, process_keyphrase, check_present_idx


def check_present_idx_debug():
    keyphrase_list = ["svm", "support", "support vector machine", ".", "in this paper", "pca", "suppor", ""]
    source = "In this paper, we propose a support vector machine to classify."
    keyphrase_list_tokenized = [get_tokens(keyphrase.lower(), True) for keyphrase in keyphrase_list]
    source_tokens = get_tokens(source.lower(), True)
    print(source_tokens)
    src_len = len(source_tokens)
    print(src_len)
    print(check_present_idx(source_tokens, keyphrase_list_tokenized))
    return


def find_variations_from_wiki_debug():
    keyphrase_list = ["svm", "support vector machine", "principal component analysis", "abcdefg", "pca"]
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True)
    for keyphrase in keyphrase_list:
        print(find_variations_from_wiki(keyphrase, source_tokens, True))
    return


def find_variations_debug():
    keyphrase_list = ["svm", "support vector machine", "principal component analysis", "abcdefg", "pca", "principal component analysis (pca)", "apple (a b c d e f g h)"]
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True)
    for keyphrase in keyphrase_list:
        print(find_variations(keyphrase, source_tokens, fine_grad=True, limit_num=True))
    return


def process_keyphrase_debug():
    keyphrase_list = ["svm", "support vector machine", "principal component analysis", "abcdefg", "pca",
                      "principal component analysis (pca)", "apple (a b c d e f g h)", ""]
    keyphrase_str = ';'.join(keyphrase_list)
    source = "In this paper, we propose support vector machines to classify."
    source_tokens = get_tokens(source.lower(), True)
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


if __name__ == '__main__':
    #process_keyphrase_debug()
    check_present_idx_debug()

