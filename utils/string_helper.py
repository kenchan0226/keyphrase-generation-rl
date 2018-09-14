from nltk.stem.porter import *
stemmer = PorterStemmer()

def prediction_to_sentence(prediction, idx2word, vocab_size, oov, eos_idx):
    """
    :param prediction: a list of 0 dim tensor
    :return: a list of words
    """
    sentence = []
    for idx in prediction:
        _idx = int(idx.item())  # convert zero dim tensor to int
        if _idx == eos_idx:  # terminate the conversion if we see a <BOS>
            break
        if _idx < vocab_size:
            word = idx2word[_idx]
        else:
            word = oov[_idx - vocab_size]
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



