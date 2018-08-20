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
* All digits should be replaced by a '<digit>' tokens.
* In source files, the title and the main body are separated by an '<eos>' token
* in target files, the keyphrases are separated by an ';' character

## Numericalize Data
The `preprocess.py` script numericalizes the three pairs of source-target files, and produce the following files
`train.one2one.pt, train.one2many.pt, valid.one2one.pt, valid.one2many.pt, test.one2one.pt, test.one2many.pt, vocab.pt`.
The `*.one2one.pt` files are used for training in a supervised learning setting using maximum likehood.
The `*.one2many.pt` files are used for training in a reinforcement learning (RL) setting using reward.

Command:
`python3 preprocess.py -data_dir data/[dataset]`

## Training
Example of training command:
`python3 train.py -data data/kp20k/ -vocab data/kp20k/ -exp_path exp/%s.%s -exp kp20k -copy_attention -coverage_attn -train_ml`

Some common options for the training script:
```
-data [Path prefix to the "train.one2one.pt" and "train.one2many.pt" file path from preprocess.py], e.g., -data data/kp20k/
-vocab [Path prefix to the "vocab.pt" file path from preprocess.py], e.g., -vocab data/kp20k/
-exp [Name of the experiment for logging.], e.g., kp20k
-exp_path [Path of experiment log/plot.], e.g., -exp_path exp/%s.%s, the %s will be filled by the value in -exp and timestamp
-copy_attention, a flag for training a model with copy attention, we follow the copy attention in [See et al. 2017]
-coverage_attn, a flag for training a model with coverage attention layer, we follow the coverage attention in [See et al. 2017]
-train_ml, a flag for training a model using maximum likehood in a supervised learning setting.
-train_ml, a flag for training a model using reward in a reinforcement learning setting.
```
Please read the config.py for more details about the options.

## TODO
- [ ] Beam Search
- [x] Early Stopping of training
- [ ] Evaluation
- [ ] Training with RL

## References
Abigail See, Peter J. Liu, Christopher D. Manning:
Get To The Point: Summarization with Pointer-Generator Networks. ACL (1) 2017: 1073-1083

Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky, Yu Chi:
Deep Keyphrase Generation. ACL (1) 2017: 582-592
