# Deep Keypharse Generation with one-to-many
Our implementation is built on the source code from [seq2seq-keyphrase-pytorch](https://github.com/memray/seq2seq-keyphrase-pytorch), which is the official implementation of the Deep Keyphrase Generation paper \[Meng et al. 2017\].

## Dependencies
* python 3.5+
* pytorch 0.4

## Dataset
For a new dataset, you should create a new directory and place it in the `./data` directory.
The directory should contains three pairs of source-target files, `train_src.txt, train_trg.txt, valid_src.txt, valid_trg.txt, test_src.txt`.
The `./data` directory contains the `kp20k_small` folder, it only contains a small dataset for debugging.
For the full kp20k dataset, please download from here [here](https://www.dropbox.com/s/b5zudclq0pfjdor/kp20k.zip?dl=0). Thanks to Mr. Wang Chen for his help on data preprocessing.

### Formats
* All the text should be tokenized and all the tokens should be separated by a space character.
* All digits should be replaced by a `<digit>` tokens.
* In source files, the title and the main body are separated by an `<eos>` token
* in target files, the keyphrases are separated by an `;` character

## Numericalize Data
The `preprocess.py` script numericalizes the three pairs of source-target files, and produce the following files
`train.one2one.pt, train.one2many.pt, valid.one2one.pt, valid.one2many.pt, test.one2one.pt, test.one2many.pt, vocab.pt`.
The `*.one2one.pt` files are used for training in a supervised learning setting using maximum likehood.
The `*.one2many.pt` files are used for training in a reinforcement learning (RL) setting using reward.

Command:
`python3 preprocess.py -data_dir data/[dataset]`

## Training
Command example 1 (one2one baseline):

`python3 train.py -data data/kp20k/ -vocab data/kp20k/ -exp_path exp/%s.%s -exp kp20k -copy_attention -train_ml`

Command example 2 (one2many baseline, concatenated all the predicted keyphrases for one src, keyphrases are separated by `<sep>` token):

`python3 train.py -data data/kp20k/ -vocab data/kp20k/ -exp_path exp/%s.%s -exp kp20k -copy_attention -train_ml -one2many -one2many_mode 1 -delimiter_type 0 -batch_size 8`

Command example 3 (one2many baseline, reset the hidden state after the prediction of each keyphrase, keyphrases are separated by `<eos>` token):

`python3 train.py -data data/kp20k/ -vocab data/kp20k/ -exp_path exp/%s.%s -exp kp20k -copy_attention -train_ml -one2many -one2many_mode 1 -delimiter_type 1 -batch_size 8`

Some common options for the training script:
```
-data [Path prefix to the "train.one2one.pt" and "train.one2many.pt" file path from preprocess.py], e.g., -data data/kp20k/
-vocab [Path prefix to the "vocab.pt" file path from preprocess.py], e.g., -vocab data/kp20k/
-exp [Name of the experiment for logging.], e.g., kp20k
-exp_path [Path of experiment log/plot.], e.g., -exp_path exp/%s.%s, the %s will be filled by the value in -exp and timestamp
-copy_attention, a flag for training a model with copy attention, we follow the copy attention in [See et al. 2017]
-coverage_attn, a flag for training a model with coverage attention layer, we follow the coverage attention in [See et al. 2017]
-coverage_loss, a flag for training a model with coverage loss in [See et al. 2017]
-lambda_coverage [coefficient of coverage loss], a coefficient to control the importance of coverage loss, default is 1.
-train_ml, a flag for training a model using maximum likehood in a supervised learning setting.
-train_rl, a flag for training a model using reward in a reinforcement learning setting.
-one2many, a flag for training a model using one2many mode.
-one2many_mode [mode], [mode]=1: concatenated the keyphrases by <sep>; [mode]=2: reset the inital state after each keyphrases.
-delimiter_type [type], only effective in one2many mode. If [type] = 0, SEP_WORD=<sep>, if [type] = 1, SEP_WORD=<eos>. Default is 1.
```
Please read the config.py for more details about the options.

## Testing
Examples of testing commands:

First, run the predict.py to output all the predicted keyphrases to a text file.

`python3 predict.py -data data/kp20k/ -vocab data/kp20k/ -exp_path exp/%s.%s -exp kp20k -pred_path pred/%s.%s -enc_layers 2 -batch_size 8 -beam_size 100 -copy_attention -model [path_to_saved_model]`
After that, it create a predict.txt in the path specified by pred_path, e.g., pred/predict.kp20k.bi-directional.20180914-095220/predictions.txt.
For each line in the prediction.txt contains all the predicted keyphrases for a source. The keyphrases are separated by ';'.

Then, run the evaluate_prediction.py to compute the evaluation metric
`python3 evaluate_prediction.py -pred_file_path pred/predict.kp20k.bi-directional.20180916-152826/predictions.txt -src_file_path data/kp20k/test_src.txt -trg_file_path data/kp20k/test_trg.txt -exp_path exp/predict.kp20k.bi-directional.20180916-152826 -filtered_pred_path pred/predict.kp20k.bi-directional.20180916-152826 -exp kp20k -export_filtered_pred`

The options for evaluate_prediction.py:
```
-pred_file_path [path of the file exported by predict.py]
-src_file_path [path of the source file in the dataset], e.g., data/kp20k/test_src.txt
-trg_file_path [path of the target file in the dataset], e.g., data/kp20k/test_trg.txt
-exp_path [path for experiment log, which includes all the evaluation results]
-exp [Name of the experiment for logging.], e.g., kp20k
-export_filtered_pred, a flag for exporting all the filtered keyphrases to a file
-filtered_pred_path [path of the file that store the filtered keyphrases]
```

## TODO
- [x] Beam Search
- [x] Early Stopping of training
- [x] Evaluation
- [ ] Adjust the code and parameters of the baseline until we get competitive results
- [ ] Training with RL

## References
Abigail See, Peter J. Liu, Christopher D. Manning:
Get To The Point: Summarization with Pointer-Generator Networks. ACL (1) 2017: 1073-1083

Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky, Yu Chi:
Deep Keyphrase Generation. ACL (1) 2017: 582-592
