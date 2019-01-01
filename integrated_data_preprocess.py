import os
import json
import re
import string
import argparse
import wikipedia
import requests
import json
from evaluate_prediction import check_present_and_duplicate_keyphrases, check_duplicate_keyphrases
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from utils import string_helper
import numpy as np

DIGIT = '<digit>'
KEYWORDS_TUNCATE = 10
MAX_KEYWORD_LEN = 6
PRINTABLE = set(string.printable)
# CoreNLP = StanfordCoreNLP(r'/research/king3/hpchan/stanford-corenlp-full-2017-06-09')
CoreNLP = StanfordCoreNLP(r'/nlp/CoreNLP/stanford-corenlp-full-2018-02-27/')

def check_present_idx(src_str, keyphrase_str_list):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return: an np array that stores the keyphrase's start idx in the src if it present in src. else, the value is len(src) +1
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)
    src_len = len(src_str)

    present_indices = np.ones(num_keyphrases) * (src_len+1)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)
        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string, treat it as absent
            present_indices[i] = src_len + 1
        else:
            # check if it appears in source text
            match = False
            for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
                match = True
                for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                    src_w = src_str[src_start_idx + keyphrase_i]
                    if src_w != keyphrase_w:
                        match = False
                        break
                if match:
                    present_indices[i] = src_start_idx
                    break
            if not match:
                present_indices[i] = src_len + 1

    return present_indices


def find_variations(keyphrase, src_tokens, fine_grad, limit_num):
    """
    :param keyphrase: must be stripped
    :param src_tokens: tokenized src, a list of words
    :return: a string that contains all the variations of a given keyphrase, the variations are separated by '|', e.g., 'v11 v12|v21 v22'
    """

    if keyphrase == "":
        return ""

    keyphrase_variations = []
    # insert the acronym as one of the variations if there is a () place at the end of the keyphrase
    if keyphrase[-1] == ')':
        match_list = re.findall('\((.*?)\)', keyphrase)  # match all str inside parenthesis
        if len(match_list) > 0:
            acronym = match_list[-1]  # the last match should be an acronym
            keyphrase_variations.append(get_tokens(acronym, fine_grad))
    # remove the parenthesis and insert the keyphrase as one of the variations
    keyphrase_filtered = re.sub(r'\(.*?\)', '', keyphrase).strip()

    # debug, if after filtering, keyphrase becomes empty:
    if keyphrase_filtered == "":
        # If the keyphrase becomes empty after removing parenthesis, replace with the value inside the paraenthesis.
        keyphrase_filtered = acronym
        print("Keyphrase becomes empty after removing parenthesis")
        print("From {} to {}.".format(keyphrase, keyphrase_filtered))
        #print(keyphrase)
        #exit()

    keyphrase_variations.append(get_tokens(keyphrase_filtered, fine_grad))
    # find variations from wikipedia
    keyphrase_variations += find_variations_from_wiki(keyphrase_filtered, src_tokens, fine_grad)  # a list of word list
    # remove duplicates
    # keyphrase_variations contains the original keyphrase, the text within a () in the original keyphrase if any, and the variations from wiki
    # we need to remove duplicate variations, i.e., we only keep the variations that have unique word stems
    # first stem the variations, then remove the duplicates
    keyphrase_variations_stemmed = string_helper.stem_str_list(keyphrase_variations)  # a list of word list
    not_duplicate = check_duplicate_keyphrases(keyphrase_variations_stemmed)  # a boolean np array
    keyphrase_variations_unique = [' '.join(v) for v, is_keep in zip(keyphrase_variations, not_duplicate) if is_keep and (not limit_num or len(v) <= MAX_KEYWORD_LEN)]  # ['v11 v12', 'v21 v22']
    return '|'.join(keyphrase_variations_unique)  # 'v11 v12|v21 v22'


def find_variations_from_wiki(keyphrase, src_tokens, fine_grad):
    """
    :param phrase:
    :param src_str: tokenized source
    :return: a list of tokenized phrase variations, contains the title of the entity as well as the titles that redirected to the entities
    """
    wiki_variations = []
    try:
        entity = wikipedia.page(title=keyphrase, auto_suggest=False, redirect=True)
        entity_title = entity.title  # without lowercase
        #entity_title_tokens = get_tokens(entity.title.lower(), fine_grad)  # lowercase and tokenize
        stage = 1

    except wikipedia.exceptions.DisambiguationError as e:
        stage = 2
        possible_titles = e.options  # fetch all the possible entity titles, a list of str, without lowercase
        # lowercase and then tokenize possible titles, ignore it if a possible title is an empty string
        possible_titles_tokenized = [get_tokens(title.lower(), fine_grad) for title in possible_titles if title.strip()!= '']  # a list of word lists
        # stem possible titles
        possible_titles_stemmed = string_helper.stem_str_list(possible_titles_tokenized)  # a list of word lists
        # stem src
        src_stemmed = string_helper.stem_word_list(src_tokens)  # word list
        is_present, not_duplicate = check_present_and_duplicate_keyphrases(src_stemmed, possible_titles_stemmed)
        possible_titles_that_present_in_src = [title for title, is_keep in zip(possible_titles, is_present) if is_keep]
        if len(possible_titles_that_present_in_src) == 0:
            return []
        else:
            entity_title = possible_titles_that_present_in_src[0]

    except wikipedia.exceptions.PageError as e:
        return []
    except wikipedia.exceptions.WikipediaException as e:
        print(keyphrase)
        print(e)
        exit()
    except KeyError as e:
        print(keyphrase)
        if e.args[0] == 'pages':
            return []
        else:
            print(e)
            exit()
    except Exception as e:  # catch *all* exceptions
        print(keyphrase)
        print(e)
        exit()

    if entity_title == "":
        print("Entity title is empty!")
        print(keyphrase)
        if stage == 2:
            print(possible_titles)
            print(possible_titles_that_present_in_src)
        exit()

    entity_title_tokens = get_tokens(entity_title.lower(), fine_grad)  # lowercase and tokenize
    wiki_variations.append(entity_title_tokens)
    wiki_variations += find_redirected_titles(entity_title, fine_grad)  # a list of word list
    # wiki_variations contains the title of the entity as well as the titles that redirected to the entities
    return wiki_variations


def find_redirected_titles(entity_title, fine_grad):
    """
    :param entity_title: without lowercase
    :return: titles_that_redicted_to_the_entity: a list of list of words, tokenized
    """
    # find all the names that are redirected to this entity
    url = "http://en.wikipedia.org/w/api.php?action=query&list=backlinks&bltitle={}&blfilterredir=redirects&format=json".format(
        entity_title)
    response = requests.get(url)
    response_json = json.loads(response.text)
    # lowercase and remove the parenthesis in the titles that are redirected to the entitiy, and then tokenize it
    try:
        titles_that_redirected_to_the_entity = [get_tokens(re.sub(r'\(.*?\)', "", entry['title'].lower()).strip(), fine_grad) for
                                            entry in response_json['query']['backlinks']]  # a list of word list
    except KeyError as e:
        print(e)
        print(entity_title)
        print(url)
        print(response_json)
        exit()
    return titles_that_redirected_to_the_entity


def get_tokens(text, fine_grad=True):
    """
    Need use the same word tokenizer between keywords and source context
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    """
    text = re.sub(r'[\r\n\t]', '', text)
    text = ''.join(list(filter(lambda x: x in PRINTABLE, text)))
    if fine_grad:
        # tokenize by non-letters
        # Although we have will use corenlp for tokenizing,
        # we still use the following tokenizer for fine granularity
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
    else:
        tokens = text.split()
    # replace the digit terms with <digit>
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]
    # tokens = CoreNLP.word_tokenize(' '.join(tokens))
    # c = ' '.join(CoreNLP.word_tokenize(c.strip())) + '\n'

    return tokens


def process_keyphrase(keyword_str, src_tokens, variations=True, limit_num=True, fine_grad=True):
    # remove question mark
    keyword_str = keyword_str.replace('?', '')
    # remove the any '[' or ']' symbol
    keyword_str = keyword_str.replace('[', '')
    keyword_str = keyword_str.replace(']', '')
    keyphrase_list = []
    for keyphrase in keyword_str.split(';'):
        keyphrase = keyphrase.strip()
        if len(keyphrase) > 0:  # if keyphrase is not an empty string
            if variations:
                keyphrase_variations = find_variations(keyphrase, src_tokens, fine_grad, limit_num)  # str of variations, e.g., 'v11 v12|v21 v22'
                if len(keyphrase_variations) > 0:
                    keyphrase_list.append(keyphrase_variations)
            else:
                keyphrase = re.sub(r'\(.*?\)', '', keyphrase)  # remove text in parenthesis
                # tokenize, then serialize and add to the keyphrase_list if it does not exceed MAX_KEYWORD_LEN
                keyphrase_tokens = get_tokens(keyphrase.strip(), fine_grad)  # word list
                if len(keyphrase_tokens) == 0:
                    continue
                elif limit_num and len(keyphrase_tokens) > MAX_KEYWORD_LEN:
                    continue
                else:
                    keyphrase = ' '.join(keyphrase_tokens)  # a keyphrase str, e.g., 'k11 k12'
                    keyphrase_list.append(keyphrase)
    # a list of keyphrase str
    return keyphrase_list


def json2txt_for_corenlp(json_home, dataset, data_type, saved_home, fine_grad=True, use_orig_keys=False, variations=False):
    """
    process the original json file into a txt file for corenlp tokenizing
    :param json_home: the home directory of the json files of KP20k
    :param data_type: training, testing, validation
    :param saved_home: the directory to save the obtained txt file
    :param use_orig_keys: Whether directly use the original keys (unprocessed).
    :return: None
    """
    print('\nProcessing {} data...'.format(data_type))
    saved_data_dir = os.path.join(saved_home, 'data_for_corenlp')
    if not os.path.exists(saved_data_dir):
        os.makedirs(saved_data_dir)
    json_file_name = os.path.join(json_home, "{}_{}.json".format(dataset, data_type))
    json_file = open(json_file_name, encoding='utf-8')
    processed_keyword_file = open(os.path.join(saved_data_dir, "{}_{}_keyword_for_corenlp.txt".format(dataset, data_type)),
                                  'w', encoding='utf-8')
    # context = title + '.' + '<eos>' + abstract
    processed_context_file = open(os.path.join(saved_data_dir, "{}_{}_context_for_corenlp.txt".format(dataset, data_type)),
                                  'w', encoding='utf-8')
    lines = json_file.readlines()
    for line_idx in tqdm(range(len(lines))):
        line = lines[line_idx]
        line_dict = json.loads(line.strip())
        # tokenization, lowercasing, replace all digits with '<digit>' symbol
        title = line_dict['title'].strip()
        abstract = line_dict['abstract'].strip()
        keywords = line_dict['keyword'].strip()
        # lowercasing the text
        title = title.lower()
        abstract = abstract.lower()
        keywords = keywords.lower()
        # filter out no-title or no-abstract data
        if len(title) == 0 or len(abstract) == 0:
            continue
        # concatenate title and abstract
        context = title + ' . ' + ' <eos> ' + abstract

        # for fine granularity tokenization
        context_tokens = get_tokens(context, fine_grad=fine_grad)
        context = ' '.join(context_tokens)
        if not use_orig_keys:
            if data_type != 'testing':
                # keyword_str, src_tokens, find_variations=True, limit_num=True, fine_grad=True
                keywords = ';'.join(process_keyphrase(keywords, context_tokens, variations=variations, limit_num=True, fine_grad=fine_grad))
            else:
                keywords = ';'.join(process_keyphrase(keywords, context_tokens, variations=variations, limit_num=False, fine_grad=fine_grad))
        else:
            keywords = ';'.join(keywords.strip().split(';'))

        context_line = context + '\n'
        keywords_line = keywords + '\n'

        processed_keyword_file.write(keywords_line)
        processed_context_file.write(context_line)

    processed_keyword_file.close()
    processed_context_file.close()


def filter_dups(saved_home, dups_info_home):
    """
    filter out the duplicates in the training data with the testing data according to the obtained duplication info file.
    :param saved_home: non-filtered data home
    :param dups_info_home: duplication information home
    :return: None
    """
    orig_context_file = open(os.path.join(saved_home, 'data_for_corenlp', 'kp20k_training_context_for_corenlp.txt'),
                             encoding='utf-8')
    context_lines = orig_context_file.readlines()
    orig_allkeys_file = open(os.path.join(saved_home, 'data_for_corenlp', 'kp20k_training_keyword_for_corenlp.txt'),
                             encoding='utf-8')
    allkeys_lines = orig_allkeys_file.readlines()
    assert len(context_lines) == len(allkeys_lines)

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
                                              'kp20k_training_context_for_corenlp_filtered.txt'),
                                 'w', encoding='utf-8')
    filtered_context_file.writelines(context_lines)

    filtered_allkeys_file = open(os.path.join(saved_home, 'data_for_corenlp',
                                              'kp20k_training_keyword_for_corenlp_filtered.txt'),
                                 'w', encoding='utf-8')
    filtered_allkeys_file.writelines(allkeys_lines)

    orig_context_file = open(os.path.join(saved_home, 'data_for_corenlp',
                                          'kp20k_training_filtered_for_corenlp_idxes.txt'),
                             'w', encoding='utf-8')
    orig_context_file.write(' '.join([str(idx) for idx in total_filtered_idxes]) + '\n')
    orig_context_file.write(str(len(total_filtered_idxes)) + '\n')


def corenlp_tokenizing(data_home, dataset='kp20k', data_type='validation'):
    """
    Use corenlp to tokenize the text
    Corenlp Installation: https://github.com/Lynten/stanford-corenlp
    :param data_for_corenlp_home: the directory for the original data
    :param dataset: dataset name ['kp20k', 'inspec', 'krapivin', 'nus', 'semeval']
    :param data_type: ['training', 'validation', 'testing']
    :return: None
    """
    suffix = ''
    if dataset == 'kp20k' and data_type == 'training':
        suffix = '_filtered'
    data_for_opennmt_home = os.path.join(data_home, 'data_for_opennmt')
    if not os.path.exists(data_for_opennmt_home):
        os.makedirs(data_for_opennmt_home)

    context_file = os.path.join(data_home, 'data_for_corenlp', '{}_{}_context_for_corenlp{}.txt'.format(dataset, data_type, suffix))
    context_file = open(context_file, encoding='utf-8')
    context_lines = context_file.readlines()
    # tokenized_context_lines = [' '.join(CoreNLP.word_tokenize(c.strip())) + '\n' for c in context_lines]
    tokenized_context_lines = []
    for c_idx in tqdm(range(len(context_lines))):
        c = context_lines[c_idx]
        c = ' '.join(CoreNLP.word_tokenize(c.strip())) + '\n'
        tokenized_context_lines.append(c)
    saved_context_file = os.path.join(data_for_opennmt_home, '{}_{}_context{}.txt'.format(dataset, data_type, suffix))
    saved_context_file = open(saved_context_file, 'w', encoding='utf-8')
    saved_context_file.writelines(tokenized_context_lines)

    key_file = os.path.join(data_home, 'data_for_corenlp', '{}_{}_keyword_for_corenlp{}.txt'.format(dataset, data_type, suffix))
    key_file = open(key_file, encoding='utf-8')
    key_lines = key_file.readlines()
    # tokenized_key_lines = [' '.join(CoreNLP.word_tokenize(c.strip())) + '\n' for c in key_lines]
    tokenized_key_lines = []
    for c_idx in tqdm(range(len(key_lines))):
        c = key_lines[c_idx]
        c = ' '.join(CoreNLP.word_tokenize(c.strip())) + '\n'
        tokenized_key_lines.append(c)
    saved_key_file = os.path.join(data_for_opennmt_home, '{}_{}_keyword{}.txt'.format(dataset, data_type, suffix))
    saved_key_file = open(saved_key_file, 'w', encoding='utf-8')
    saved_key_file.writelines(tokenized_key_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='integrated_data_preprocess')
    parser.add_argument('-json_home', type=str,
                        default='process_json/integrated_processed_data/json_home')
    parser.add_argument('-saved_home', type=str,
                        default='process_json/integrated_processed_data')
    parser.add_argument('-dups_info_home', type=str,
                        default='process_json/duplicates_w_kp20k_training')
    parser.add_argument('-dataset', type=str, default='kp20k',
                        choices=['kp20k', 'inspec', 'krapivin', 'nus', 'semeval'])
    parser.add_argument('-data_type', type=str, default='validation', choices=['training', 'validation', 'testing'])
    parser.add_argument('-fine_grad', action='store_true',
                        help='Whether tokenizing the text in the style of RuiMeng before using cornlp tokenizing')
    #parser.add_argument('-lowercase', action='store_true',
    #                    help='Whether lowercasing all the text')
    parser.add_argument('-use_orig_keys', action='store_true',
                        help='Whether directly use the original keys (unprocessed).')
    parser.add_argument('-variations', action='store_true',
                        help='Whether to enrich the keyphrases with their variations from wikipeida.')
    opts = parser.parse_args()

    # 1. convert the json file into txt file w/
    # lowercasing (if needed), RuiMeng's tokenizing (if needed),
    # replacing digits with <digit>, filtering out the data with empty title or abstract,
    # filtering out the keyphrases which have more than MAX_KEYWORD_LEN tokens
    #
    # Note: no keyphrase number truncation in this script.
    #
    # set -fine_grad; -use_orig_keys
    #
    json2txt_for_corenlp(json_home=opts.json_home, dataset=opts.dataset, data_type=opts.data_type, saved_home=opts.saved_home,
                         fine_grad=opts.fine_grad, use_orig_keys=opts.use_orig_keys, variations=opts.variations)

    # 2. filter out the duplicates in the kp20k training data
    # filter_dups(saved_home=opts.saved_home, dups_info_home=opts.dups_info_home)

    # 3. tokenize text using corenlp
    # corenlp_tokenizing(data_home=opts.saved_home, dataset=opts.dataset, data_type=opts.data_type)
