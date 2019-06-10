import logging
import os
import sys
import time

def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger

def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=100,
                        help='Word embedding for both.')

    #parser.add_argument('-position_encoding', action='store_true',
    #                    help='Use a sin to mark relative words positions.')
    parser.add_argument('-share_embeddings', default=True, action='store_true',
                        help="""Share the word embeddings between encoder
                         and decoder.""")
    parser.add_argument('-use_target_encoder', action='store_true',
                        help="Use target decoder")

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='rnn',
                        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-encoder_size', type=int, default=150,
                        help='Size of encoder hidden states')
    parser.add_argument('-decoder_size', type=int, default=300,
                        help='Size of decoder hidden states')
    parser.add_argument('-target_encoder_size', type=int, default=64,
                        help='Size of target encoder hidden states')
    parser.add_argument('-source_representation_queue_size', type=int, default=128,
                        help='Size of queue for storing the encoder representation for training the target encoder')
    parser.add_argument('-source_representation_sample_size', type=int, default=32,
                        help='Sample size of encoder representation for training the target encoder.')
    parser.add_argument('-dropout', type=float, default=0.1,
                        help="Dropout probability; applied in LSTM stacks.")
    # parser.add_argument('-input_feed', type=int, default=1,
    #                     help="""Feed the context vector at each time step as
    #                     additional input (via concatenation with the word
    #                     embeddings) to the decoder.""")

    #parser.add_argument('-rnn_type', type=str, default='GRU',
    #                    choices=['LSTM', 'GRU'],
    #                    help="""The gate type to use in the RNNs""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    #parser.add_argument('-input_feeding', action="store_true",
    #                    help="Apply input feeding or not. Feed the updated hidden vector (after attention)"
    #                         "as new hidden vector to the decoder (Luong et al. 2015). "
    #                         "Feed the context vector at each time step  after normal attention"
    #                         "as additional input (via concatenation with the word"
    #                         "embeddings) to the decoder.")

    parser.add_argument('-bidirectional', default=True,
                        action = "store_true",
                        help="whether the encoder is bidirectional")

    parser.add_argument('-bridge', type=str, default='copy',
                        choices=['copy', 'dense', 'dense_nonlinear', 'none'],
                        help="An additional layer between the encoder and the decoder")

    # Attention options
    parser.add_argument('-attn_mode', type=str, default='concat',
                       choices=['general', 'concat'],
                       help="""The attention type to use:
                       dot or general (Luong) or concat (Bahdanau)""")
    #parser.add_argument('-attention_mode', type=str, default='concat',
    #                    choices=['dot', 'general', 'concat'],
    #                    help="""The attention type to use:
    #                    dot or general (Luong) or concat (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('-copy_attention', action="store_true",
                        help='Train a copy model.')

    #parser.add_argument('-copy_mode', type=str, default='concat',
    #                    choices=['dot', 'general', 'concat'],
    #                    help="""The attention type to use: dot or general (Luong) or concat (Bahdanau)""")

    #parser.add_argument('-copy_input_feeding', action="store_true",
    #                    help="Feed the context vector at each time step after copy attention"
    #                         "as additional input (via concatenation with the word"
    #                         "embeddings) to the decoder.")

    #parser.add_argument('-reuse_copy_attn', action="store_true",
    #                   help="Reuse standard attention for copy (see See et al.)")

    #parser.add_argument('-copy_gate', action="store_true",
    #                    help="A gate controling the flow from generative model and copy model (see See et al.)")

    parser.add_argument('-coverage_attn', action="store_true",
                        help='Train a coverage attention layer.')
    parser.add_argument('-review_attn', action="store_true",
                        help='Train a review attention layer')

    parser.add_argument('-lambda_coverage', type=float, default=1,
                        help='Lambda value for coverage by See et al.')
    parser.add_argument('-coverage_loss', action="store_true", default=False,
                        help='whether to include coverage loss')
    parser.add_argument('-orthogonal_loss', action="store_true", default=False,
                        help='whether to include orthogonal loss')
    parser.add_argument('-lambda_orthogonal', type=float, default=0.03,
                        help='Lambda value for the orthogonal loss by Yuan et al.')
    parser.add_argument('-lambda_target_encoder', type=float, default=0.03,
                        help='Lambda value for the target encoder loss by Yuan et al.')

    parser.add_argument('-separate_present_absent', action="store_true", default=False,
                        help='whether to separate present keyphrase predictions and absnet keyphrase predictions as two sub-tasks')
    parser.add_argument('-manager_mode', type=int, default=1, choices=[1],
                        help='Only effective in separate_present_absent. 1: two trainable vectors as the goal vectors;')
    parser.add_argument('-goal_vector_size', type=int, default=16,
                        help='size of goal vector')
    parser.add_argument('-goal_vector_mode', type=int, default=0, choices=[0, 1, 2],
                        help='Only effective in separate_present_absent. 0: no goal vector; 1: goal vector act as an extra input to the decoder; 2: goal vector act as an extra input to p_gen')
    parser.add_argument('-title_guided', action="store_true", default=False,
                        help='whether to use title-guided encoder')


    # parser.add_argument('-context_gate', type=str, default=None,
    #                     choices=['source', 'target', 'both'],
    #                     help="""Type of context gate to use.
    #                     Do not select for no context gate by Tu:2017:TACL.""")

    # group.add_argument('-lambda_coverage', type=float, default=1,
    #                    help='Lambda value for coverage.')

    # Cascading model options
    #parser.add_argument('-cascading_model', action="store_true", help='Train a copy model.')


def vocab_opts(parser):
    # Dictionary Options
    parser.add_argument('-vocab_size', type=int, default=50002,
                        help="Size of the source vocabulary")
    # for copy model
    parser.add_argument('-max_unk_words', type=int, default=1000,
                        help="Maximum number of unknown words the model supports (mainly for masking in loss)")

    parser.add_argument('-words_min_frequency', type=int, default=0)

    # Options most relevant to summarization
    parser.add_argument('-dynamic_dict', default=True,
                        action='store_true', help="Create dynamic dictionaries (for copy)")

def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the "train.one2one.pt" and
                        "train.one2many.pt" file path from preprocess.py""")
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the "vocab.pt"
                        file path from preprocess.py""")

    parser.add_argument('-custom_data_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-custom_vocab_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-vocab_filename_suffix', default='',
                        help='')
    parser.add_argument('-data_filename_suffix', default='',
                        help='')

    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    #parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
    #                    help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    # Fixed word vectors
    parser.add_argument('-fix_word_vecs_enc',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-fix_word_vecs_dec',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')
    parser.add_argument('-optim', default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=1,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-truncated_decoder', type=int, default=0,
                        help="""Truncated bptt.""")
    parser.add_argument('-loss_normalization', default="tokens", choices=['tokens', 'batches'],
                        help="Normalize the cross-entropy loss by the number of tokens or batch size")

    # Learning options
    parser.add_argument('-train_ml', action="store_true", default=False,
                        help='Train with Maximum Likelihood or not')
    parser.add_argument('-train_rl', action="store_true", default=False,
                        help='Train with Reinforcement Learning or not')

    # Reinforcement Learning options
    #parser.add_argument('-rl_method', default=0, type=int,
    #                    help="""0: ori, 1: running average as baseline""")
    parser.add_argument('-max_sample_length', default=6, type=int,
                        help="The max length of sequence that can be sampled by the model")
    parser.add_argument('-max_length', type=int, default=6,
                        help='Maximum prediction length.')
    parser.add_argument('-topk', type=str, default='M',
                        help='The only pick the top k predictions in reward.')
    parser.add_argument('-reward_type', default='0', type=int,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        help="""Type of reward. 0: f1, 1: recall, 2: ndcg, 3: accuracy, 4: alpha-ndcg, 5: alpha-dcg, 6: AP, 7: F1 penalize duplicate""")
    parser.add_argument('-match_type', default='exact',
                        choices=['exact', 'sub'],
                        help="""Either exact matching or substring matching.""")
    parser.add_argument('-pretrained_model', default="",
                        help="The path of pretrained model. Only effective in RL")
    parser.add_argument('-reward_shaping', action="store_true", default=False,
                        help="Use reward shaping in RL training")
    parser.add_argument('-baseline', default="self", choices=["none", "self"],
                        help="The baseline in RL training. none: no baseline; self: use greedy decoding as baseline")
    parser.add_argument('-mc_rollouts', action="store_true", default=False,
                        help="Use Monte Carlo rollouts to estimate q value. Not support yet.")
    parser.add_argument('-num_rollouts', type=int, default=3,
                        help="The number of Monte Carlo rollouts. Only effective when mc_rollouts is True. Not supported yet")

    # One2many options
    parser.add_argument('-delimiter_type', type=int, default=0, choices=[0, 1],
                        help='If type is 0, use <sep> to separate keyphrases. If type is 1, use <eos> to separate keyphrases')
    parser.add_argument('-one2many', action="store_true", default=False,
                        help='If true, it will not split a sample into multiple src-keyphrase pairs')
    parser.add_argument('-one2many_mode', type=int, default=0, choices=[1, 2, 3],
                        help='Only effective when one2many=True. 1: concatenated the keyphrases by <sep>; 2: reset the inital state and input after each keyphrase; 3: reset the input after each keyphrase')
    parser.add_argument('-num_predictions', type=int, default=1,
                        help='Control the number of predictions when one2many_mode=2. If you set the one2many_mode to 1, the number of predictions should also be 1.')

    #parser.add_argument('-loss_scale', type=float, default=0.5,
    #                    help='A scaling factor to merge the loss of ML and RL parts: L_mixed = γ * L_rl + (1 − γ) * L_ml'
    #                         'The γ used by Metamind is 0.9984 in "A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION"'
    #                         'The α used by Google is 0.017 in "Google Translation": O_Mixed(θ) = α ∗ O_ML(θ) + O_RL(θ)'
    #                     )

    #parser.add_argument('-rl_start_epoch', default=2, type=int,
    #                    help="""from which epoch rl training starts""")
    parser.add_argument('-init_perturb_std', type=float, default=0,
                        help="Init std of gaussian perturbation vector to the hidden state of the GRU after generated each a keyphrase")
    parser.add_argument('-final_perturb_std', type=float, default=0,
                        help="Final std of gaussian perturbation vector to the hidden state of the GRU after generated each a keyphrase. Only effective when perturb_decay=1")
    parser.add_argument('-perturb_decay_mode', type=int, default=1, choices=[0, 1, 2],
                        help='Specify how the std of perturbation vector decay. 0: no decay, 1: exponential decay, 2: iteration-wise decay')
    parser.add_argument('-perturb_decay_factor', type=float, default=0.0001,
                        help="Specify the decay factor, only effective when perturb_decay=1 or 2")
    parser.add_argument('-perturb_baseline', action="store_true", default=False,
                        help="Whether to perturb the baseline or not")
    #parser.add_argument('-perturb_decay_along_phrases', action="store_true", default=False,
    #                    help="Decay the perturbations along the predicted keyphrases, std=std/num_of_preds")
    parser.add_argument('-regularization_type', type=int, default=0, choices=[0, 1, 2],
                        help='0: no regularization, 1: percentage of unique keyphrases, 2: entropy')
    parser.add_argument('-regularization_factor', type=float, default=0.0,
                        help="Factor of regularization")
    parser.add_argument('-replace_unk', action="store_true",
                        help='Replace the unk token with the token of highest attention score.')
    parser.add_argument('-remove_src_eos', action="store_true",
                        help='Remove the eos token at the end of src text')

    # GPU

    # Teacher Forcing and Scheduled Sampling
    parser.add_argument('-must_teacher_forcing', action="store_true",
                        help="Apply must_teacher_forcing or not")
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0,
                        help="The ratio to apply teaching forcing ratio (default 0)")
    parser.add_argument('-scheduled_sampling', action="store_true",
                        help="Apply scheduled sampling or not")
    parser.add_argument('-scheduled_sampling_batches', type=int, default=10000,
                        help="The maximum number of batches to apply scheduled sampling")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_rl', type=float, default=0.00005,
                        help="""Starting learning rate for Reinforcement Learning.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay_rl', action="store_true", default=False,
                        help="""A flag to use learning rate decay in rl training""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=2,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")
    parser.add_argument('-checkpoint_interval', type=int, default=4000,
                        help='Run validation and save model parameters at this interval.')
    #parser.add_argument('-run_valid_every', type=int, default=4000,
    #                    help="Run validation test at this interval (every run_valid_every batches)")
    parser.add_argument('-disable_early_stop_rl', action="store_true", default=False,
                        help="A flag to disable early stopping in rl training.")
    parser.add_argument('-early_stop_tolerance', type=int, default=4,
                        help="Stop training if it doesn't improve any more for several rounds of validation")

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")

    #parser.add_argument('-save_model_every', type=int, default=2000,
    #                    help="Save checkpoint at this interval.")

    parser.add_argument('-report_every', type=int, default=10,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")
    parser.add_argument('-model_path', type=str, default="model/%s.%s",
                        help="Path of checkpoints.")

    # beam search setting
    '''
    parser.add_argument('-beam_search_batch_example', type=int, default=8,
                        help='Maximum of examples for one batch, should be disabled for training')
    parser.add_argument('-beam_search_batch_size', type=int, default=8,
                        help='Maximum batch size')
    parser.add_argument('-beam_search_batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')

    parser.add_argument('-beam_size',  type=int, default=150,
                        help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=6,
                        help='Maximum sentence length.')
    '''

def predict_opts(parser):
    parser.add_argument('-model', required=True,
                       help='Path to model .pt file')
    parser.add_argument('-verbose', action="store_true", help="Whether to log the results of every individual samples")
    parser.add_argument('-attn_debug', action="store_true", help="Whether to print attn for each word")
    #parser.add_argument('-present_kp_only', action="store_true", help="Only consider the keyphrases that present in the source text")
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the "test.one2many.pt" file path from preprocess.py""")
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the "vocab.pt"
                            file path from preprocess.py""")
    parser.add_argument('-custom_data_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-custom_vocab_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-vocab_filename_suffix', default='',
                        help='')
    parser.add_argument('-data_filename_suffix', default='',
                        help='')

    parser.add_argument('-beam_size', type=int, default=50,
                       help='Beam size')
    parser.add_argument('-n_best', type=int, default=-1,
                        help='Pick the top n_best sequences from beam_search, if n_best < 0, then n_best=beam_size')
    parser.add_argument('-max_length', type=int, default=6,
                       help='Maximum prediction length.')
    parser.add_argument('-length_penalty_factor', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
    parser.add_argument('-coverage_penalty_factor', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    parser.add_argument('-length_penalty', default='none', choices=['none', 'wu', 'avg'],
    help="""Length Penalty to use.""")
    parser.add_argument('-coverage_penalty', default='none', choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")

    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    # parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
    #                    help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                            reproducibility.""")
    parser.add_argument('-batch_size', type=int, default=8,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")

    parser.add_argument('-include_attn_dist', action="store_true",
                        help="Whether to return the attention distribution, for the visualization of the attention weights, haven't implemented")
    parser.add_argument('-pred_file_prefix', type=str, default="",
                        help="Prefix of prediction file.")
    parser.add_argument('-pred_path', type=str, default="pred/%s.%s",
                        help="Path of outputs of predictions.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")
    parser.add_argument('-one2many', action="store_true", default=False,
                        help='If true, it will not split a sample into multiple src-keyphrase pairs')
    #parser.add_argument('-greedy', action="store_true", default=False,
    #                    help='Use greedy decoding instead of sampling in one2many mode')
    parser.add_argument('-one2many_mode', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Only effective when one2many=True. 0 is a dummy option which takes no effect. 1: concatenated the keyphrases by <sep>; 2: reset the inital state and input after each keyphrase; 3: reset the input after each keyphrase')
    parser.add_argument('-delimiter_type', type=int, default=0, choices=[0, 1],
                        help='If type is 0, use <sep> to separate keyphrases. If type is 1, use <eos> to separate keyphrases')
    #parser.add_argument('-num_predictions', type=int, default=1,
    #                    help='Control the number of predictions when one2many_mode=2.')
    parser.add_argument('-max_eos_per_output_seq', type=int, default=1, # max_eos_per_seq
                        help='Specify the max number of eos in one output sequences to control the number of keyphrases in one output sequence. Only effective when one2many_mode=3 or one2many_mode=2.')
    parser.add_argument('-sampling', action="store_true",
                        help='Use sampling instead of beam search to generate the predictions.')
    parser.add_argument('-replace_unk', action="store_true",
                        help='Replace the unk token with the token of highest attention score.')
    parser.add_argument('-remove_src_eos', action="store_true",
                        help='Remove the eos token at the end of src text')
    parser.add_argument('-block_ngram_repeat', type=int, default=0,
                        help='Block repeat of n-gram')
    parser.add_argument('-ignore_when_blocking', nargs='+', type=str,
                        default=['<sep>'],
                        help="""Ignore these strings when blocking repeats.
                               You want to block sentence delimiters.""")


def post_predict_opts(parser):
    parser.add_argument('-pred_file_path', type=str, required=True,
                        help="Path of the prediction file.")
    parser.add_argument('-src_file_path', type=str, required=True,
                        help="Path of the source text file.")
    parser.add_argument('-trg_file_path', type=str,
                        help="Path of the target text file.")
    parser.add_argument('-export_filtered_pred', action="store_true",
                        help="Export the filtered predictions to a file or not")
    parser.add_argument('-filtered_pred_path', type=str, default="",
                        help="Path of the folder for storing the filtered prediction")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="",
                        help="Path of experiment log/plot.")
    parser.add_argument('-disable_extra_one_word_filter', action="store_true",
                        help="If False, it will only keep the first one-word prediction")
    parser.add_argument('-disable_valid_filter', action="store_true",
                        help="If False, it will remove all the invalid predictions")
    parser.add_argument('-num_preds', type=int, default=200,
                        help='It will only consider the first num_preds keyphrases in each line of the prediction file')
    parser.add_argument('-debug', action="store_true", default=False,
                        help='Print out the metric at each step or not')
    parser.add_argument('-match_by_str', action="store_true", default=False,
                        help='If false, match the words at word level when checking present keyphrase. Else, match the words at string level.')
    parser.add_argument('-invalidate_unk', action="store_true", default=False,
                        help='Treat unk as invalid output')
    parser.add_argument('-target_separated', action="store_true", default=False,
                        help='The targets has already been separated into present keyphrases and absent keyphrases')
    parser.add_argument('-prediction_separated', action="store_true", default=False,
                        help='The predictions has already been separated into present keyphrases and absent keyphrases')
    parser.add_argument('-reverse_sorting', action="store_true", default=False,
                        help='Only effective in target separated.')
    parser.add_argument('-tune_f1_v', action="store_true", default=False,
                        help='For tuning the F1@V score.')
    parser.add_argument('-all_ks', nargs='+', default=['5', '10', 'M'], type=str,
                        help='only allow integer or M')
    parser.add_argument('-present_ks', nargs='+', default=['5', '10', 'M'], type=str,
                        help='')
    parser.add_argument('-absent_ks', nargs='+', default=['5', '10', '50', 'M'], type=str,
                        help='')
    parser.add_argument('-target_already_stemmed', action="store_true", default=False,
                        help='If it is true, it will not stem the target keyphrases.')
    parser.add_argument('-meng_rui_precision', action="store_true", default=False,
                        help='If it is true, when computing precision, it will divided by the number pf predictions, instead of divided by k.')
    parser.add_argument('-use_name_variations', action="store_true", default=False,
                        help='Match the ground-truth with name variations.')

def interactive_predict_opts(parser):
    parser.add_argument('-model', required=True,
                       help='Path to model .pt file')
    parser.add_argument('-attn_debug', action="store_true", help="Whether to print attn for each word")
    parser.add_argument('-src_file', required=True,
                        help="""Path to source file""")
    #parser.add_argument('-trg_file', required=True,
    #                    help="""Path to target file""")
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the "vocab.pt"
                            file path from preprocess.py""")
    parser.add_argument('-custom_vocab_filename_suffix', action="store_true",
                        help='')
    parser.add_argument('-vocab_filename_suffix', default='',
                        help='')
    parser.add_argument('-beam_size', type=int, default=50,
                       help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help='Pick the top n_best sequences from beam_search, if n_best < 0, then n_best=beam_size')
    parser.add_argument('-max_length', type=int, default=60,
                       help='Maximum prediction length.')
    parser.add_argument('-length_penalty_factor', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
    parser.add_argument('-coverage_penalty_factor', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    parser.add_argument('-length_penalty', default='none', choices=['none', 'wu', 'avg'],
    help="""Length Penalty to use.""")
    parser.add_argument('-coverage_penalty', default='none', choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    parser.add_argument('-gpuid', default=0, type=int,
                        help="Use CUDA on the selected device.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                            reproducibility.""")
    parser.add_argument('-batch_size', type=int, default=8,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=1,
                        help='Number of workers for generating batches')

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="The current time stamp.")

    parser.add_argument('-include_attn_dist', action="store_true",
                        help="Whether to return the attention distribution, for the visualization of the attention weights, haven't implemented")

    parser.add_argument('-pred_path', type=str, required=True,
                        help="Path of outputs of predictions.")
    parser.add_argument('-pred_file_prefix', type=str, default="",
                        help="Prefix of prediction file.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    #parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
    #                    help="Path of experiment log/plot.")
    parser.add_argument('-one2many', action="store_true", default=False,
                        help='If true, it will not split a sample into multiple src-keyphrase pairs')
    #parser.add_argument('-greedy', action="store_true", default=False,
    #                    help='Use greedy decoding instead of sampling in one2many mode')
    parser.add_argument('-one2many_mode', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Only effective when one2many=True. 0 is a dummy option which takes no effect. 1: concatenated the keyphrases by <sep>; 2: reset the inital state and input after each keyphrase; 3: reset the input after each keyphrase')
    parser.add_argument('-delimiter_type', type=int, default=0, choices=[0, 1],
                        help='If type is 0, use <sep> to separate keyphrases. If type is 1, use <eos> to separate keyphrases')
    parser.add_argument('-max_eos_per_output_seq', type=int, default=1,  # max_eos_per_seq
                        help='Specify the max number of eos in one output sequences to control the number of keyphrases in one output sequence. Only effective when one2many_mode=3 or one2many_mode=2.')
    parser.add_argument('-sampling', action="store_true",
                        help='Use sampling instead of beam search to generate the predictions.')
    parser.add_argument('-replace_unk', action="store_true",
                            help='Replace the unk token with the token of highest attention score.')
    parser.add_argument('-remove_src_eos', action="store_true",
                        help='Remove the eos token at the end of src text')
    parser.add_argument('-remove_title_eos', action="store_true", default=False,
                        help='Remove the eos token at the end of title')
    parser.add_argument('-block_ngram_repeat', type=int, default=0,
                        help='Block repeat of n-gram')
    parser.add_argument('-ignore_when_blocking', nargs='+', type=str,
                       default=['<sep>'],
                       help="""Ignore these strings when blocking repeats.
                           You want to block sentence delimiters.""")
