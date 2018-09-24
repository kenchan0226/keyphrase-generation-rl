from nltk.stem.porter import *
stemmer = PorterStemmer()

def prediction_to_sentence(prediction, idx2word, vocab_size, oov, eos_idx):
    """
    :param prediction: a list of 0 dim tensor
    :return: a list of words, does not include the final EOS
    """
    sentence = []
    for i, pred in enumerate(prediction):
        _pred = int(pred.item())  # convert zero dim tensor to int
        if i == len(prediction) - 1 and _pred == eos_idx:  # ignore the final EOS token
            break
        if _pred < vocab_size:
            word = idx2word[_pred]
        else:
            word = oov[_pred - vocab_size]
        sentence.append(word)

    return sentence

def stem_str_list(str_list):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_list = []
    for word_list in str_list:
        stemmed_word_list = stem_word_list(word_list)
        stemmed_str_list.append(stemmed_word_list)
    return stemmed_str_list

def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]

def split_concated_keyphrases(word_list, delimiter_word):
    """
    :param word_list: word list of concated keyprhases, separated by a delimiter
    :param delimiter_word
    :return: a list of keyphrases from a concated sequence, each keyphrase is a word list
    """
    tmp_pred_str_list = []
    tmp_word_list = []
    for word in word_list:
        if word != delimiter_word:
            tmp_word_list.append(word)
        else:
            if len(tmp_word_list) > 0:
                tmp_pred_str_list.append(tmp_word_list)
                tmp_word_list = []
    if len(tmp_word_list) > 0:  # append the final keyphrase to the pred_str_list
        tmp_pred_str_list.append(tmp_word_list)
    return tmp_pred_str_list

