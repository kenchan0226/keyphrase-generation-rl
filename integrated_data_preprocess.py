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
import time

DIGIT = '<digit>'
KEYWORDS_TUNCATE = 10
MAX_KEYWORD_LEN = 6
PRINTABLE = set(string.printable)
FILE_NUM = {'inspec': 500, 'krapivin': 400, 'nus': 211, 'semeval': 100}


def batch_check_present_idx_backup(src_str, keyphrase_str_list):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return: an np array that stores the keyphrase's start idx in the src if it present in src. else, the value is len(src) +1
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)
    src_len = len(src_str)
    num_present_keyphrases = 0

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
                    num_present_keyphrases += 1
                    break
            if not match:
                present_indices[i] = src_len + 1

    return present_indices, num_present_keyphrases


def batch_check_present_idx(src_str, keyphrase_str_list):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return: an np array that stores the keyphrase's start idx in the src if it present in src. else, the value is len(src) +1
    """
    num_keyphrases = len(keyphrase_str_list)
    src_len = len(src_str)
    num_present_keyphrases = 0

    present_indices = np.ones(num_keyphrases) * (src_len+1)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        present_indices[i], is_present = check_present_idx(src_str, keyphrase_word_list)
        if is_present:
            num_present_keyphrases += 1

    return present_indices, num_present_keyphrases


def check_present_idx(src_str, keyphrase_word_list):
    src_len = len(src_str)
    joined_keyphrase_str = ' '.join(keyphrase_word_list)
    if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string, treat it as absent
        return src_len + 1, False
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
                present_index = src_start_idx
                break
        if not match:
            present_index = src_len + 1
    return present_index, match


def find_variations(keyphrase, src_tokens, fine_grad, limit_num, match_ending_parenthesis, use_corenlp, find_redirections):
    """
    :param keyphrase: must be stripped
    :param src_tokens: tokenized src, a list of words
    :return: a list of keyphrase variations
    """

    extract_acronym_flag = False
    acronym_tokens = None

    if keyphrase == "":
        return ""

    keyphrase_variations = []
    # insert the acronym as one of the variations if there is a () place at the end of the keyphrase
    if keyphrase[-1] == ')':
        match_list = re.findall('\((.*?)\)', keyphrase)  # match all str inside parenthesis
        if len(match_list) > 0:
            acronym = match_list[-1]  # the last match should be an acronym
            if match_ending_parenthesis and len(acronym) > 1:
                #keyphrase_variations.append(get_tokens(acronym, fine_grad, use_corenlp))
                acronym_tokens = get_tokens(acronym, fine_grad, use_corenlp)
                extract_acronym_flag = True
            #else:
            #    acronym_tokens = None
    # remove the parenthesis and insert the keyphrase as one of the variations
    keyphrase_filtered = re.sub(r'\(.*?\)', '', keyphrase).strip()

    # debug, if after filtering, keyphrase becomes empty:
    if keyphrase_filtered == "":
        # If the keyphrase becomes empty after removing parenthesis, replace with the value inside the paraenthesis.
        keyphrase_filtered = acronym
        extract_acronym_flag = False
        acronym_tokens = None
        #print("Keyphrase becomes empty after removing parenthesis")
        #print("From {} to {}.".format(keyphrase, keyphrase_filtered))
        #print(keyphrase)
        #exit()

    keyphrase_variations.append(get_tokens(keyphrase_filtered, fine_grad, use_corenlp))
    if acronym_tokens is not None:
        keyphrase_variations.append(acronym_tokens)
    # find variations from wikipedia, wiki_variations: a list of word list
    wiki_variations, num_matched_disambiguation, num_redirections_found = find_variations_from_wiki(keyphrase_filtered, src_tokens, fine_grad, use_corenlp, find_redirections)
    keyphrase_variations += wiki_variations
    # remove duplicates
    # keyphrase_variations contains the original keyphrase, the text within a () in the original keyphrase if any, and the variations from wiki
    # we need to remove duplicate variations, i.e., we only keep the variations that have unique word stems
    # first stem the variations, then remove the duplicates
    keyphrase_variations_stemmed = string_helper.stem_str_list(keyphrase_variations)  # a list of word list
    not_duplicate = check_duplicate_keyphrases(keyphrase_variations_stemmed)  # a boolean np array
    keyphrase_variations_unique = [' '.join(v) for v, is_keep in zip(keyphrase_variations, not_duplicate) if is_keep and (not limit_num or len(v) <= MAX_KEYWORD_LEN)]  # ['v11 v12', 'v21 v22']
    return keyphrase_variations_unique, num_matched_disambiguation, extract_acronym_flag, num_redirections_found  # 'v11 v12|v21 v22'
    #return '|'.join(keyphrase_variations_unique), match_disambiguation_flag, extract_acronym_flag  # 'v11 v12|v21 v22'


def find_variations_from_wiki(keyphrase, src_tokens, fine_grad, use_corenlp, find_redirections):
    """
    :param phrase:
    :param src_str: tokenized source
    :return: a list of tokenized phrase variations, contains the title of the entity as well as the titles that redirected to the entities; a flag for indicating that we find a disambiguation that match the source str
    """
    wiki_variations = []
    num_matched_disambiguation = 0
    num_redirections_found = 0

    max_retry = 100
    retry_flag = False
    for retry_i in range(max_retry):
        try:
            entity = wikipedia.page(title=keyphrase, auto_suggest=False, redirect=True)
            entity_title = entity.title  # without lowercase
            #entity_title_tokens = get_tokens(entity.title.lower(), fine_grad)  # lowercase and tokenize
            stage = 1
            retry_flag = False

        except wikipedia.exceptions.DisambiguationError as e:
            stage = 2
            possible_titles = e.options  # fetch all the possible entity titles, a list of str, without lowercase
            # lowercase and then tokenize possible titles, ignore it if a possible title is an empty string
            possible_titles_tokenized = [get_tokens(title.lower(), fine_grad, use_corenlp) for title in possible_titles if title.strip()!= '']  # a list of word lists
            # stem possible titles
            possible_titles_stemmed = string_helper.stem_str_list(possible_titles_tokenized)  # a list of word lists
            # stem src
            src_stemmed = string_helper.stem_word_list(src_tokens)  # word list
            is_present, not_duplicate = check_present_and_duplicate_keyphrases(src_stemmed, possible_titles_stemmed)
            possible_titles_that_present_in_src = [title for title, is_keep in zip(possible_titles, is_present) if is_keep]
            num_matched_disambiguation = len(possible_titles_that_present_in_src)
            if num_matched_disambiguation == 0:
                return [], num_matched_disambiguation, num_redirections_found
            else:
                entity_title = possible_titles_that_present_in_src[0]
                retry_flag = False
        except wikipedia.exceptions.PageError as e:
            return [], num_matched_disambiguation, num_redirections_found
        except wikipedia.exceptions.HTTPTimeoutError as e:
            if retry_i == max_retry - 1:
                raise ValueError("HTTP time out for {} times, still cannot call wikipedia API".format(max_retry))
            retry_flag = True
            time.sleep(10)
        except wikipedia.exceptions.WikipediaException as e:
            print(keyphrase)
            print(e)
            print("base exceptions")
            if retry_i == max_retry - 1:
                raise ValueError("Retry for {} times, still cannot call wikipedia API".format(max_retry))
            retry_flag = True
            time.sleep(10)
        except KeyError as e:
            return [], num_matched_disambiguation, num_redirections_found
            """
            if e.args[0] == 'pages':
                return []
            else:
                print(e)
                exit()
            """
        except Exception as e:  # catch *all* exceptions
            print(keyphrase)
            print(e)
            print("all exceptions")
            if retry_i == max_retry - 1:
                raise ValueError("Retry for {} times, still cannot call wikipedia API".format(max_retry))
            retry_flag = True
            time.sleep(10)

        if entity_title == "":
            print("Entity title is empty!")
            print(keyphrase)
            if stage == 2:
                print(possible_titles)
                print(possible_titles_that_present_in_src)
            exit()

        if not retry_flag:
            break

    entity_title_tokens = get_tokens(entity_title.lower(), fine_grad, use_corenlp)  # lowercase and tokenize
    wiki_variations.append(entity_title_tokens)
    if find_redirections:
        titles_that_redirected_to_the_entity = find_redirected_titles(entity_title, fine_grad, use_corenlp)  # a list of word list
        num_redirections_found = len(titles_that_redirected_to_the_entity)
        wiki_variations += titles_that_redirected_to_the_entity
        #wiki_variations += find_redirected_titles(entity_title, fine_grad, use_corenlp)  # a list of word list
    # wiki_variations contains the title of the entity as well as the titles that redirected to the entities
    return wiki_variations, num_matched_disambiguation, num_redirections_found


def find_redirected_titles(entity_title, fine_grad, use_corenlp):
    """
    :param entity_title: without lowercase
    :return: titles_that_redicted_to_the_entity: a list of list of words, tokenized
    """
    # find all the names that are redirected to this entity
    url = "http://en.wikipedia.org/w/api.php?action=query&list=backlinks&bltitle={}&blfilterredir=redirects&format=json".format(
        entity_title)
    max_retry = 100
    for i in range(max_retry):
        try:
            response = requests.get(url)
            break
        except requests.ConnectionError:
            if i == max_retry - 1:
                raise ValueError("Retry for {} times, still cannot get the redirected titles".format(max_retry))
            time.sleep(10)

    response_json = json.loads(response.text)
    # lowercase and remove the parenthesis in the titles that are redirected to the entitiy, and then tokenize it
    try:
        titles_that_redirected_to_the_entity = [get_tokens(re.sub(r'\(.*?\)', "", entry['title'].lower()).strip(), fine_grad, use_corenlp) for
                                            entry in response_json['query']['backlinks']]  # a list of word list
    except KeyError as e:
        print(e)
        print(entity_title)
        print(url)
        print(response_json)
        exit()
    return titles_that_redirected_to_the_entity


def get_tokens(text, fine_grad=True, use_corenlp=True):
    """
    Need use the same word tokenizer between keywords and source context
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    """
    if replace_with_space:
        text = re.sub(r'[\r\n\t]', ' ', text)
    else:
        text = re.sub(r'[\r\n\t]', '', text)
    text = ''.join(list(filter(lambda x: x in PRINTABLE, text)))
    if fine_grad:
        # tokenize by non-letters
        # Although we have will use corenlp for tokenizing,
        # we still use the following tokenizer for fine granularity
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
    else:
        tokens = text.split()

    if use_corenlp:
        tokens = CoreNLP.word_tokenize(' '.join(tokens))
    # c = ' '.join(CoreNLP.word_tokenize(c.strip())) + '\n'

    # replace the digit terms with <digit>
    if fine_grad_digit_matching:
        tokens = [w if not re.match('^[+-]?((\d+(\.\d*)?)|(\.\d+))$', w) else DIGIT for w in tokens]
    else:
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens


def remove_duplicate_from_str_list(str_list):
    unique_str_list = []
    str_set = set()
    for a_str in str_list:
        if a_str not in str_set:
            str_set.add(a_str)
            unique_str_list.append(a_str)
    return unique_str_list


def process_keyphrase(keyword_str, src_tokens, keyphrase_stat, variations=False, limit_num=True, fine_grad=True, sort_keyphrases=False, match_ending_parenthesis=False, use_corenlp=True, separate_present_absent=False, find_redirections=False):
    if variations and sort_keyphrases:
        raise ValueError("You cannot use sort_keyphrases when you need to find the variations of each keyphrase")
    # remove question mark
    #keyword_str = keyword_str.replace('?', '')
    # remove the any '[' or ']' symbol
    #keyword_str = keyword_str.replace('[', '')
    #keyword_str = keyword_str.replace(']', '')
    #keyword_str = keyword_str.replace('|', '')
    # remove '?', '[', ']', '|', '\\' characters
    keyword_str = re.sub(r'[\\|\[\]?]', '', keyword_str)
    keyphrase_list = []
    keyphrase_token_2dlist = []
    for keyphrase in keyword_str.split(';'):
        keyphrase = keyphrase.strip()
        if len(keyphrase) > 0:  # if keyphrase is not an empty string
            keyphrase_stat['num_keyphrases'] += 1
            if variations:
                keyphrase_variations, num_matched_disambiguation, extract_acronym_flag, num_redirections_found = find_variations(keyphrase, src_tokens, fine_grad, limit_num, match_ending_parenthesis, use_corenlp, find_redirections)  # str of variations, e.g., 'v11 v12|v21 v22'
                keyphrase_variations_str = '|'.join(keyphrase_variations) # serialize it into a string, each variation is separated by '|', e.g., 'v11 v12|v21 v22'
                if len(keyphrase_variations) > 0:
                    keyphrase_list.append(keyphrase_variations_str)
                    keyphrase_stat['num_variations'] += len(keyphrase_variations)
                    if num_matched_disambiguation > 0:
                        keyphrase_stat['num_matched_disambiguation'] += num_matched_disambiguation
                        keyphrase_stat['num_keyphrases_with_match_disambiguation'] += 1
                    if extract_acronym_flag:
                        keyphrase_stat['num_extracted_acronym'] += 1
                    if len(keyphrase_variations) > 1:
                        keyphrase_stat['num_keyphrases_with_variations'] += 1
                    if num_redirections_found > 0:
                        keyphrase_stat['num_keyphrases_with_redirections'] += 1
                        keyphrase_stat['num_redirections'] += num_redirections_found

            else:
                keyphrase_filtered = re.sub(r'\(.*?\)', '', keyphrase).strip()  # remove text in parenthesis
                if keyphrase_filtered == "":  # if keyphrase is empty after removing parenthesis, just keep the text inside the parenthesis
                    match_list = re.findall('\((.*?)\)', keyphrase)  # match all str inside parenthesis
                    keyphrase = match_list[-1]
                else:
                    keyphrase = keyphrase_filtered
                # tokenize, then serialize and add to the keyphrase_list if it does not exceed MAX_KEYWORD_LEN
                keyphrase_tokens = get_tokens(keyphrase.strip(), fine_grad, use_corenlp)  # word list
                if len(keyphrase_tokens) == 0:
                    continue
                elif limit_num and len(keyphrase_tokens) > MAX_KEYWORD_LEN:
                    continue
                else:
                    keyphrase_token_2dlist.append(keyphrase_tokens)
                    keyphrase = ' '.join(keyphrase_tokens)  # a keyphrase str, e.g., 'k11 k12'
                    keyphrase_list.append(keyphrase)

    if sort_keyphrases:
        keyphrase_list = sort_keyphrases_by_their_order_of_occurence(keyphrase_list, src_tokens, keyphrase_token_2dlist, separate_present_absent)

    # remove duplicate keyphrases
    keyphrase_list = remove_duplicate_from_str_list(keyphrase_list)

    # a list of keyphrase str
    return keyphrase_list


def sort_keyphrases_by_their_order_of_occurence(keyphrase_list, src_tokens, keyphrase_token_2dlist, separate_present_absent):
    num_keyphrase = len(keyphrase_list)
    assert num_keyphrase == len(keyphrase_token_2dlist)
    # stem the token list and check the present idx
    src_tokens_stemmed = string_helper.stem_word_list(src_tokens)
    keyphrase_token_2dlist_stemmed = string_helper.stem_str_list(keyphrase_token_2dlist)
    present_idx_array, num_present_keyphrases = batch_check_present_idx(src_tokens_stemmed, keyphrase_token_2dlist_stemmed)
    # rearrange the order in keyphrase list
    sorted_keyphrase_indices = np.argsort(present_idx_array)
    sorted_keyphrase_list = [keyphrase_list[idx] for idx in sorted_keyphrase_indices]
    if separate_present_absent:
        if reverse_sorting:
            sorted_keyphrase_list = sorted_keyphrase_list[num_present_keyphrases:] + [present_absent_segmenter] + sorted_keyphrase_list[:num_present_keyphrases]
        else:
            sorted_keyphrase_list.insert(num_present_keyphrases, present_absent_segmenter)
    return sorted_keyphrase_list
    #return [keyphrase_list[idx] for idx in sorted_keyphrase_indices]


def process_cross_doamin_file(home_folder, dataset, saved_home, fine_grad=True, variations=False, sort_keyphrases=False, match_ending_parenthesis=False, use_corenlp=True, separate_present_absent=False, find_redirections=False):
    processed_files_suffix = ""
    if variations:
        processed_files_suffix += "_variations"
    if find_redirections:
        processed_files_suffix += "_redirections"
    if sort_keyphrases:
        processed_files_suffix += "_sorted"
    if match_ending_parenthesis:
        processed_files_suffix += "_parenthesis"
    if separate_present_absent:
        processed_files_suffix += "_separated"
    if fine_grad_digit_matching:
        processed_files_suffix += "_digit"
    if reverse_sorting:
        processed_files_suffix += "_reversed"
    if replace_with_space:
        processed_files_suffix += "_space"

    context_file_path = os.path.join(saved_home, 'data_for_corenlp', '{}_testing_context_for_corenlp{}.txt'.format(dataset, processed_files_suffix))
    trg_file_path = os.path.join(saved_home, 'data_for_corenlp', '{}_testing_keyword_for_corenlp{}.txt'.format(dataset, processed_files_suffix))
    keywords_file = open(trg_file_path, 'w')
    context_file = open(context_file_path, 'w')
    keywords_lines = []
    context_lines = []
    keyphrase_stat = {'num_keyphrases_with_variations': 0, 'num_keyphrases': 0, 'num_variations': 0,
                      'num_keyphrases_with_match_disambiguation': 0, 'num_extracted_acronym': 0,
                      'num_keyphrases_with_redirections': 0,
                      'num_redirections': 0, 'num_matched_disambiguation': 0}

    file_num = FILE_NUM[dataset]
    for i in tqdm(range(file_num)):
        keywords_file_i = open( os.path.join(home_folder, dataset, 'keyphrase', '{}.txt'.format(i)) )
        context_file_i = open( os.path.join(home_folder, dataset, 'text', '{}.txt'.format(i)) )

        context_i_line = context_file_i.readlines()[0]
        context_i_line = [w.split('_')[0] for w in context_i_line.strip().split()]

        context_i_tokens = get_tokens(' '.join(context_i_line), fine_grad=fine_grad, use_corenlp=use_corenlp)
        context_i_line = ' '.join(context_i_tokens) + '\n'

        keywords_i = [line.strip() for line in keywords_file_i.readlines()]
        keywords_i_line = ';'.join(keywords_i)
        keywords_i_line = ';'.join(
                process_keyphrase(keywords_i_line, context_i_tokens, keyphrase_stat, variations=variations, limit_num=False,
                                  fine_grad=fine_grad, sort_keyphrases=sort_keyphrases,
                                  match_ending_parenthesis=match_ending_parenthesis, use_corenlp=use_corenlp,
                                  separate_present_absent=separate_present_absent, find_redirections=find_redirections)) + '\n'

        if dataset != 'krapivin':
            context_i_line = context_i_line.replace('<eos>', '. <eos>')

        keywords_lines.append(keywords_i_line)
        context_lines.append(context_i_line)

    keywords_file.writelines(keywords_lines)
    context_file.writelines(context_lines)

    return


def json2txt_for_corenlp(json_home, dataset, data_type, saved_home, fine_grad=True, use_orig_keys=False, variations=False, sort_keyphrases=False, match_ending_parenthesis=False, use_corenlp=True, separate_present_absent=False, find_redirections=False):
    """
    process the original json file into a txt file for corenlp tokenizing
    :param json_home: the home directory of the json files of KP20k
    :param data_type: training, testing, validation
    :param saved_home: the directory to save the obtained txt file
    :param use_orig_keys: Whether directly use the original keys (unprocessed).
    :return: None
    """
    keyphrase_stat = {'num_keyphrases_with_variations': 0, 'num_keyphrases': 0, 'num_variations': 0,
                      'num_keyphrases_with_match_disambiguation': 0, 'num_extracted_acronym': 0, 'num_keyphrases_with_redirections': 0,
                      'num_redirections': 0, 'num_matched_disambiguation': 0}
    #num_keyphrases_with_variations = 0
    #num_keyphrases = 0
    #num_variations = 0
    #num_keyphrases_with_match_disambiguation = 0

    if variations and sort_keyphrases:
        raise ValueError("You cannot use sort_keyphrases when you need to find the variations of each keyphrase")

    print('\nProcessing {} data...'.format(data_type))
    saved_data_dir = os.path.join(saved_home, 'data_for_corenlp')
    if not os.path.exists(saved_data_dir):
        os.makedirs(saved_data_dir)
    json_file_name = os.path.join(json_home, "{}_{}.json".format(dataset, data_type))
    json_file = open(json_file_name, encoding='utf-8')
    processed_files_suffix = ""
    if variations:
        processed_files_suffix += "_variations"
    if find_redirections:
        processed_files_suffix += "_redirections"
    if sort_keyphrases:
        processed_files_suffix += "_sorted"
    if match_ending_parenthesis:
        processed_files_suffix += "_parenthesis"
    if separate_present_absent:
        processed_files_suffix += "_separated"
    if fine_grad_digit_matching:
        processed_files_suffix += "_digit"
    if reverse_sorting:
        processed_files_suffix += "_reversed"
    if replace_with_space:
        processed_files_suffix += "_space"

    processed_keyword_file = open(os.path.join(saved_data_dir, "{}_{}_keyword_for_corenlp{}.txt".format(dataset, data_type, processed_files_suffix)),
                                  'w', encoding='utf-8')
    # context = title + '.' + '<eos>' + abstract
    processed_context_file = open(os.path.join(saved_data_dir, "{}_{}_context_for_corenlp{}.txt".format(dataset, data_type, processed_files_suffix)),
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
        context_tokens = get_tokens(context, fine_grad=fine_grad, use_corenlp=use_corenlp)
        context = ' '.join(context_tokens)
        if not use_orig_keys:
            if data_type != 'testing':
                limit_num = True
            else:
                limit_num = False
            keywords = ';'.join(
                process_keyphrase(keywords, context_tokens, keyphrase_stat, variations=variations, limit_num=limit_num,
                                  fine_grad=fine_grad, sort_keyphrases=sort_keyphrases,
                                  match_ending_parenthesis=match_ending_parenthesis, use_corenlp=use_corenlp,
                                  separate_present_absent=separate_present_absent, find_redirections=find_redirections))
        else:
            keywords = ';'.join(keywords.strip().split(';'))

        context_line = context + '\n'
        keywords_line = keywords + '\n'

        processed_keyword_file.write(keywords_line)
        processed_context_file.write(context_line)

    processed_keyword_file.close()
    processed_context_file.close()
    print("# keyphrases: {}".format(keyphrase_stat['num_keyphrases']))
    if variations:
        print("# variations: {}".format(keyphrase_stat['num_variations']))
        print("# keyphrases with variations: {}".format(keyphrase_stat['num_keyphrases_with_variations']))
        print("# keyphrases with match disambiguation: {}".format(keyphrase_stat['num_keyphrases_with_match_disambiguation']))
        print("# matched disambiguation: {}".format(keyphrase_stat['num_matched_disambiguation']))
    if match_ending_parenthesis:
        print("# extracted acronyms: {}".format(keyphrase_stat['num_extracted_acronym']))
    if find_redirections:
        print('# redirections found: {}'.format(keyphrase_stat['num_redirections']))
        print('# keyphrases with redirections: {}'.format(keyphrase_stat['num_keyphrases_with_redirections']))
    # keyphrase_stat = {'num_keyphrases_with_variations': 0, 'num_keyphrases': 0, 'num_variations': 0, 'num_keyphrases_with_match_disambiguation': 0}


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
    parser.add_argument('-raw_txt_home', type=str,
                        default='process_json/integrated_processed_data/cross_domain_raw_txt')
    parser.add_argument('-saved_home', type=str,
                        default='process_json/integrated_processed_data')
    parser.add_argument('-dups_info_home', type=str,
                        default='process_json/duplicates_w_kp20k_training')
    parser.add_argument('-dataset', type=str, default='kp20k',
                        choices=['kp20k', 'inspec', 'krapivin', 'nus', 'semeval'])
    parser.add_argument('-data_type', type=str, default='validation', choices=['training', 'validation', 'testing', 'debug'])
    parser.add_argument('-fine_grad', action='store_true',
                        help='Whether tokenizing the text in the style of RuiMeng before using cornlp tokenizing')
    #parser.add_argument('-lowercase', action='store_true',
    #                    help='Whether lowercasing all the text')
    parser.add_argument('-use_orig_keys', action='store_true',
                        help='Whether directly use the original keys (unprocessed).')
    parser.add_argument('-variations', action='store_true',
                        help='Whether to enrich the keyphrases with their variations from wikipeida.')
    parser.add_argument('-sort_keyphrases', action='store_true',
                        help='Whether to sort the keyphrases according to their first occurrence in the source.')
    parser.add_argument('-match_ending_parenthesis', action='store_true',
                        help='Whether to extract an acronym from the ending parenthesis.')
    parser.add_argument('-use_corenlp', action='store_true',
                        help='Whether to use stanford corenlp tokenizing')
    parser.add_argument('-corenlp_home', type=str, default='/nlp/CoreNLP/stanford-corenlp-full-2018-02-27/',
                        help='Whether to use stanford corenlp tokenizing')
    parser.add_argument('-separate_present_absent', action='store_true',
                        help='Whether to separate present and absent keyphrase using another token.')
    parser.add_argument('-find_redirections', action='store_true',
                        help='Whether to enrich the keyphrases with the redirections from wikipeida.')
    parser.add_argument('-fine_grad_digit_matching', action='store_true',
                        help='Whether to use fine grad digit replace.')
    parser.add_argument('-replace_with_space', action='store_true',
                        help='Replace \t \n by a space')
    parser.add_argument('-reverse_sorting', action='store_true',
                        help='Reverse the order of sorting, only effective in sort_keyphrase')

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

    if opts.use_corenlp:
        # CoreNLP = StanfordCoreNLP(r'/research/king3/hpchan/stanford-corenlp-full-2016-10-31')
        # CoreNLP = StanfordCoreNLP(r'/nlp/CoreNLP/stanford-corenlp-full-2018-02-27/')
        CoreNLP = StanfordCoreNLP(r'{}'.format(opts.corenlp_home))

    if opts.match_ending_parenthesis:
        ending_parenthesis_output_path = os.path.join(opts.json_home, "{}_{}_ending_parenthesis_output.txt".format(opts.dataset, opts.data_type))
        # processed_keyword_file = open(os.path.join(saved_data_dir, "{}_{}_keyword_for_corenlp.txt".format(dataset, data_type)), 'w', encoding='utf-8')

    if opts.separate_present_absent:
        present_absent_segmenter = "<peos>"
        if not opts.sort_keyphrases:
            raise ValueError("If you want to separate present keyphrase and basent keyphrase, you must specify the option -sort_keyphrases.")

    if opts.fine_grad_digit_matching:
        fine_grad_digit_matching = True
    else:
        fine_grad_digit_matching = False

    if opts.replace_with_space:
        replace_with_space = True
    else:
        replace_with_space = False

    if opts.reverse_sorting:
        reverse_sorting = True
    else:
        reverse_sorting = False

    if opts.dataset == "kp20k":
        json2txt_for_corenlp(json_home=opts.json_home, dataset=opts.dataset, data_type=opts.data_type, saved_home=opts.saved_home,
                             fine_grad=opts.fine_grad, use_orig_keys=opts.use_orig_keys, variations=opts.variations,
                             sort_keyphrases=opts.sort_keyphrases, match_ending_parenthesis=opts.match_ending_parenthesis,
                             use_corenlp=opts.use_corenlp, separate_present_absent=opts.separate_present_absent,
                             find_redirections=opts.find_redirections)
    else:
        process_cross_doamin_file(home_folder=opts.raw_txt_home, dataset=opts.dataset, saved_home=opts.saved_home, fine_grad=opts.fine_grad, variations=opts.variations,
                                  sort_keyphrases=opts.sort_keyphrases, match_ending_parenthesis=opts.match_ending_parenthesis, use_corenlp=opts.use_corenlp,
                                  separate_present_absent=opts.separate_present_absent, find_redirections=opts.find_redirections)

    # 2. filter out the duplicates in the kp20k training data
    # filter_dups(saved_home=opts.saved_home, dups_info_home=opts.dups_info_home)

    # 3. tokenize text using corenlp
    # corenlp_tokenizing(data_home=opts.saved_home, dataset=opts.dataset, data_type=opts.data_type)
