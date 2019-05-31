# Keyphrase Generation via Reinforcement Learning
This repository contains the code for our ACL 19 paper "Neural Keyphrase Generation via Reinforcement Learning with Adaptive Rewards".

Our implementation is built on the source code from [seq2seq-keyphrase-pytorch](https://github.com/memray/seq2seq-keyphrase-pytorch)\[Meng et al. 2017\].
Some codes are adapted from this [repository](https://github.com/atulkum/pointer_summarizer). The code for beam search algorithm is mainly adapted from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

If you use this code, please cite our paper:
```
@inproceedings{conf/acl/chan19keyphraseRL,
  title={Neural Keyphrase Generation via Reinforcement Learning with Adaptive Rewards},
  author={Hou Pong Chan, Wang Chen, Lu Wang, and Irwin King},
  booktitle={Proceedings of ACL},
  year={2019}
}
```

## Dependencies
* python 3.5+
* pytorch 0.4

## Dataset
For a new dataset, you should create a new directory and place it in the `./data` directory.
The directory should contains three pairs of source-target files, `train_src.txt, train_trg.txt, valid_src.txt, valid_trg.txt, test_src.txt`.
The `./data` directory contains the `kp20k_small` folder, it only contains a small dataset for debugging.
For the full kp20k dataset, please download from here [here](https://www.dropbox.com/s/b5zudclq0pfjdor/kp20k.zip?dl=0), which is a tokenized version of the data provided by [Rui Meng](https://github.com/memray/seq2seq-keyphrase). We also removed the duplicated documents in the training set according to their instructions.
Thanks to Mr. Wang Chen for his help on data preprocessing.

### Formats
* All the text should be tokenized and all the tokens should be separated by a space character.
* All digits should be replaced by a `<digit>` tokens.
* In source files, the title and the main body are separated by an `<eos>` token
* in target files, the keyphrases are separated by an `;` character. There is no space before and after the colon character, e.g., `keyphrase one;keyphrase two`.

## Numericalize Data
The `preprocess.py` script numericalizes the three pairs of source-target files, and produce the following files
`train.one2one.pt, train.one2many.pt, valid.one2one.pt, valid.one2many.pt, test.one2one.pt, test.one2many.pt, vocab.pt`.
The `*.one2one.pt` files which split a sample (source, {kp1, kp2, ...}) into multiple training sample (source, kp1), (source, kp2), ...
The `*.one2many.pt` files does not split the training sample.

Command:
`python3 preprocess.py -data_dir data/[dataset]`

## Cross-entropy Loss Training
Command example 1 (one2one baseline):

`python3 train.py -data data/kp20k_filtered/ -vocab data/kp20k_filtered/ -exp_path exp/%s.%s -exp kp20k_filtered -copy_attention -train_ml`

Command example 2 (one2many baseline, concatenated all the predicted keyphrases for one src, keyphrases are separated by `<sep>` token):

`python3 train.py -data data/kp20k_filtered/ -vocab data/kp20k_filtered/ -exp_path exp/%s.%s -exp kp20k_filtered -copy_attention -train_ml -one2many -one2many_mode 1 -delimiter_type 0 -batch_size 8`

Command example 3 (one2many baseline, reset the hidden state after the prediction of each keyphrase, keyphrases are separated by `<eos>` token):

`python3 train.py -data data/kp20k_filtered/ -vocab data/kp20k_filtered/ -exp_path exp/%s.%s -exp kp20k_filtered -copy_attention -train_ml -one2many -one2many_mode 1 -delimiter_type 1 -batch_size 8`

Some common options for the training script:
```
-data []: path prefix to the "train.one2one.pt" and "train.one2many.pt" file path from preprocess.py, e.g., -data data/kp20k_filtered/
-vocab []: path prefix to the "vocab.pt" file path from preprocess.py, e.g., -vocab data/kp20k_filtered/
-exp []: name of the experiment for logging., e.g., kp20k
-exp_path []: path of experiment log/plot, e.g., -exp_path exp/%s.%s, the %s will be filled by the value in -exp and timestamp
-copy_attention: a flag for training a model with copy attention, we follow the copy attention in [See et al. 2017]
-coverage_attn: a flag for training a model with coverage attention layer, we follow the coverage attention in [See et al. 2017]
-coverage_loss: a flag for training a model with coverage loss in [See et al. 2017]
-lambda_coverage [1]: Coefficient of coverage loss, a coefficient to control the importance of coverage loss.
-review_attn: use the review attention in Chen et al. 2018a
-orthogonal_loss: a flag to include orthogonal loss
-lambda_orthogonal []: Lambda value for the orthogonal loss by Yuan et al.
-use_target_encoder: Use the target encoder by Yuan et al.
-lambda_target_encoder []: Lambda value for the target encoder loss by Yuan et al.
-train_ml: a flag for training a model using maximum likehood in a supervised learning setting.
-one2many: a flag for training a model using one2many mode.
-one2many_mode [0]: 1 means concatenated the keyphrases by <sep>; 2 means follows Chen et al. 2018a; 3 means reset the hidden state whenever the decoder emits a <EOS> token.
-delimiter_type [0]: only effective in one2many mode. If delimiter_type = 0, SEP_WORD=<sep>, if delimiter_type = 1, SEP_WORD=<eos>.
-separate_present_absent: whether to separate present keyphrase predictions and absnet keyphrase predictions by a <peos> token.
-goal_vector_mode [0]: Only effective in when using separate_present_absent. 0: do not use goal vector, 1: goal vector act as an input to the decoder, 2: goal vector act as an extra input to p_gen
-goal_vector_size [16]: Size of gaol vector
-manger_mode [1]: Only effective in when using separate_present_absent. 1: two trainable vectors as the goal vectors. May support different types of maanger in the future.
```
Please read the config.py for more details about the options.

## RL training
After you pretrain a model using cross-entropy loss, you can proceed to RL training.
In this work, we use the reward obtained by greedy decoding as the baseline (self-critical policy gradient).

Command example:
`python3 train.py -data data/kp20k_filtered/ -vocab data/kp20k_filtered/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -enc_layers 1 -train_rl -copy_attention -one2many -one2many_mode 1 -delimiter_type 0 -batch_size 32 -pretrained_model model/kp20k.ml.one2many.cat.copy.bi-directional.20190115-224431/kp20k.ml.one2many.cat.copy.bi-directional.epoch=3.batch=38600.total_batch=124000.model -max_length 60 -baseline self -reward_type 7 -replace_unk`

Some common options for rl training:
```
-train_rl: a flag for training a model using reward in a reinforcement learning setting.
-baseline []: specify the baseline for the policy gradient algorithm, choices=["none", "self"], "self" means we use self-critical as the baseline
-reward_type []: 0: f1, 1: recall, 2: ndcg, 3: accuracy, 4: alpha-ndcg, 5: alpha-dcg, 6: AP, 7: F1 (all duplicates are considered as incorrect)
-pretrained_model []: path of the MLE pre-trained model
-replace_unk: replace the unk token with the token that received the highest attention score.
-max_length []: max length of the output sequence
-num_predictions []: only effective when one2many_mode=2 or 3, control the number of predicted keyphrases.
-topk [M]: only pick the -topk predictions when computing the reward. M means use all the predictions to compute the reward.
```

We can also add Guassian noise vector to perturb the hidden state of GRU after generated each a keyphrase to encourage exploration. I tried it, but the performance is not good.
The followings are the options for the perturbation.
```
-init_perturb_std [0]: initial std of gaussian noise vector
-final_perturb_std [0]: terminal std of gaussian noise vector, only effective when perturb_decay=1.
-perturb_decay [0]: mode of decay for the std of gaussian noise, 0 means no decay, 1 means exponential decay, 2 means stepwise decay.
-perturb_decay_factor [0]: factor for the std decay, the effect depends on the value of -perturb_decay.
-perturb_baseline: a flag for perturb the hidden state of the baseline in policy gradient training.
```

We decay the std of the Guassian noise vector using the following methods
- Exponential decay: <img src="https://latex.codecogs.com/gif.latex?\sigma=$\sigma_{T}&plus;(\sigma_{0}-\sigma_{T})\exp(-tk)" title="\sigma=$\sigma_{T}+(\sigma_{0}-\sigma_{T})\exp(-tk)" />,
where <img src="https://latex.codecogs.com/gif.latex?\sigma_{0}" title="\sigma_{0}" /> is the initial std, and <img src="https://latex.codecogs.com/gif.latex?\sigma_{T}" title="\sigma_{T}" /> is the terminal std, k is the decay factor, t is the number of iterations minus 1
- Iteration-wise decay: the std is multiplied by decay factor after every 4000 iterations.

We can also regularize the reward using the following two options. The baseline reward is not affected by the regularization.
```
-regularization_type []: 0 means no regulaization, 1 means using percentage of unique keyphrases as regularization, 2 means using entropy of policy as regularization
-regularization_factor []: factor of regularization, regularized reward = (1-regularization_factor)*reward + regularization_factor*regularization
```


## Testing for kp20k dataset
Examples of testing commands:

First, run the predict.py to output all the predicted keyphrases to a text file.

`python3 predict.py -data data/kp20k_filtered/ -vocab data/kp20k_filtered/ -exp_path exp/%s.%s -exp kp20k -pred_path pred/%s.%s -enc_layers 2 -batch_size 8 -beam_size 100 -copy_attention -replace_unk -model [path_to_saved_model]`
After that, it create a predict.txt in the path specified by pred_path, e.g., pred/predict.kp20k.bi-directional.20180914-095220/predictions.txt.
For each line in the prediction.txt contains all the predicted keyphrases for a source. The keyphrases are separated by ';'.

Then, run the evaluate_prediction.py to compute the evaluation metric
`python3 evaluate_prediction.py -pred_file_path pred/predict.kp20k.bi-directional.20180916-152826/predictions.txt -src_file_path data/kp20k_filtered/test_src.txt -trg_file_path data/kp20k_filtered/test_trg.txt -exp_path exp/predict.kp20k.bi-directional.20180916-152826 -filtered_pred_path pred/predict.kp20k.bi-directional.20180916-152826 -exp kp20k -export_filtered_pred -invalidate_unk -disable_extra_one_word_filter -num_preds 200`

The options for evaluate_prediction.py:
```
-pred_file_path []: path of the file exported by predict.py
-src_file_path []: path of the source file in the dataset, e.g., data/kp20k_filtered/test_src.txt
-trg_file_path []: path of the target file in the dataset, e.g., data/kp20k_filtered/test_trg.txt
-exp_path []: path for experiment log, which includes all the evaluation results
-exp []: name of the experiment for logging, e.g., kp20k
-export_filtered_pred: a flag for exporting all the filtered keyphrases to a file
-filtered_pred_path []: path of the file that store the filtered keyphrases
-invalidate_unk: filter out all the unk words in predictions before computing the scores
-disable_extra_one_word_filter: If you did not specify this option, it will only consider the first one-word prediction. Please use this option when using kp20k testing set.
-num_preds []: It will only consider the first -num_preds keyphrases in each line of the prediction file.
-replace_unk: replace the unk token with the token that received the highest attention score.
```

## Testing for cross-domain dataset
First, run the interactive_predict.py to output all the predicted keyphrases to a text file.

Example 1:
`python3 interactive_predict.py -vocab data/kp20k_filtered/ -src_file data/cross_domain/[src_file.txt] -pred_path pred/%s.%s -enc_layers 2 -copy_attention -one2many -one2many_mode 1 -delimiter 0 -model model/kp20k.rl.copy.bi-directional.20181216-174101/kp20k.rl.copy.bi-directional.epoch=9.batch=13728.total_batch=140000.model -max_length 60 -remove_title_eos -beam_size 50 -batch_size 8 -replace_unk`

Example 2:
`python3 interactive_predict.py -vocab data/kp20k_filtered/ -src_file data/cross_domain/[src_file.txt] -pred_path pred/%s.%s -enc_layers 2 -copy_attention -model model/kp20k.ml.copy.bi-directional.20181022-151326/kp20k.ml.copy.bi-directional.epoch=3.batch=35250.total_batch=112000.model -max_length 6 -remove_title_eos -beam_size 100 -batch_size 6 -replace_unk -n_best 100`

The options for interactive_predict.py:
```
-vocab []: path prefix to the "vocab.pt" file path from preprocess.py, e.g., -vocab data/kp20k_filtered/
-src_file []: path of the source file in the dataset, e.g., -src_file data/cross_domain/word_krapivin_testing_context.txt
-pred_path []: path of the prediction file, e.g., -pred_path pred/%s.%s, the %s will be filled by the value in -exp and timestamp
-replace_unk: a flag for replacing every <unk> token with the token that received the highest attention score.
```


## References
Abigail See, Peter J. Liu, Christopher D. Manning:
Get To The Point: Summarization with Pointer-Generator Networks. ACL (1) 2017: 1073-1083

Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky, Yu Chi:
Deep Keyphrase Generation. ACL (1) 2017: 582-592

Hai Ye, Lu Wang:
Semi-Supervised Learning for Neural Keyphrase Generation. EMNLP 2018a: 4142-4153

Jun Chen, Xiaoming Zhang, Yu Wu, Zhao Yan, Zhoujun Li:
Keyphrase Generation with Correlation Constraints. EMNLP 2018a: 4057-4066

Wang Chen, Yifan Gao, Jiani Zhang, Irwin King, Michael R. Lyu:
Title-Guided Encoding for Keyphrase Generation. CoRR abs/1808.08575 (2018b)

Xingdi Yuan, Tong Wang, Rui Meng, Khushboo Thaker, Daqing He, Adam Trischler:
Generating Diverse Numbers of Diverse Keyphrases. CoRR abs/1810.05241 (2018)
