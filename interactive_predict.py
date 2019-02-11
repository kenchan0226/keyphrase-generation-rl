import torch
from sequence_generator import SequenceGenerator
import config
import argparse
from preprocess import read_tokenized_src_file
from utils.data_loader import load_vocab
from pykp.io import build_interactive_predict_dataset, KeyphraseDataset
from torch.utils.data import DataLoader
import predict
import os


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    opt.exp = 'predict.' + opt.exp
    if opt.one2many:
        opt.exp += '.one2many'

    if opt.one2many_mode == 1:
        opt.exp += '.cat'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if opt.use_target_encoder:
        opt.exp += '.target_encode'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

        # fill time into the name
    if opt.pred_path.find('%s') > 0:
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    if not opt.one2many and opt.one2many_mode > 0:
        raise ValueError("You cannot choose one2many mode without the -one2many options.")

    if opt.one2many and opt.one2many_mode == 0:
        raise ValueError("If you choose one2many, you must specify the one2many mode.")

    #if opt.greedy and not opt.one2many:
    #    raise ValueError("Greedy sampling can only be used in one2many mode.")
    return opt


def main(opt):
    # load vocab
    word2idx, idx2word, vocab = load_vocab(opt)
    # load data
    # read tokenized text file and convert them to 2d list of words
    src_file = opt.src_file
    #trg_file = opt.trg_file
    #tokenized_train_pairs = read_src_and_trg_files(src_file, trg_file, is_train=False, remove_eos=opt.remove_title_eos)  # 2d list of word
    if opt.title_guided:
        tokenized_src, tokenized_title = read_tokenized_src_file(src_file, remove_eos=opt.remove_title_eos, title_guided=True)
    else:
        tokenized_src = read_tokenized_src_file(src_file, remove_eos=opt.remove_title_eos, title_guided=False)
        tokenized_title = None
    # convert the 2d list of words to a list of dictionary, with keys 'src', 'src_oov', 'trg', 'trg_copy', 'src_str', 'trg_str', 'oov_dict', 'oov_list'
    # since we don't need the targets during testing, 'trg' and 'trg_copy' are some dummy variables
    #test_one2many = build_dataset(tokenized_train_pairs, word2idx, idx2word, opt, mode="one2many", include_original=True)
    test_one2many = build_interactive_predict_dataset(tokenized_src, word2idx, idx2word, opt, tokenized_title)
    # build the data loader
    test_one2many_dataset = KeyphraseDataset(test_one2many, word2idx=word2idx, idx2word=idx2word,
                                             type='one2many', delimiter_type=opt.delimiter_type, load_train=False, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
    test_loader = DataLoader(dataset=test_one2many_dataset,
                             collate_fn=test_one2many_dataset.collate_fn_one2many,
                             num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                             shuffle=False)
    # init the pretrained model
    model = predict.init_pretrained_model(opt)

    # Print out predict path
    print("Prediction path: %s" % opt.pred_path)

    # predict the keyphrases of the src file and output it to opt.pred_path/predictions.txt
    predict.predict(test_loader, model, opt)


if __name__=='__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='interactive_predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.interactive_predict_opts(parser)
    config.model_opts(parser)
    config.vocab_opts(parser)
    opt = parser.parse_args()

    opt = process_opt(opt)

    main(opt)

