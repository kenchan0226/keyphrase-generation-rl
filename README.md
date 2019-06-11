# Keyphrase Generation via Reinforcement Learning
This repository contains the code for our ACL 19 paper "[Neural Keyphrase Generation via Reinforcement Learning with Adaptive Rewards](https://arxiv.org/abs/1906.04106)".

Our implementation is built on the source code from [seq2seq-keyphrase-pytorch](https://github.com/memray/seq2seq-keyphrase-pytorch).
Some codes are adapted from this [repository](https://github.com/atulkum/pointer_summarizer). The code for beam search is mainly adapted from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

If you use this code, please cite our paper:
```
@inproceedings{conf/acl/chan19keyphraseRL,
  title={Neural Keyphrase Generation via Reinforcement Learning with Adaptive Rewards},
  author={Hou Pong Chan and Wang Chen and Lu Wang and Irwin King},
  booktitle={Proceedings of ACL},
  year={2019}
}
```

## Dependencies
* python 3.5+
* pytorch 0.4

## Dataset
The datasets can be downloaded from [here](https://drive.google.com/open?id=1mkDOzfSXNxcItriNNrrGV-bVv25CDRJM), which are the tokenized version of the datasets provided by [Rui Meng](https://github.com/memray/seq2seq-keyphrase). 
Please unzip the files to the `./data` directory. The `kp20k_sorted` directory contains the kp20k dataset, it consists of three pairs of source-target files, `train_src.txt, train_trg.txt, valid_src.txt, valid_trg.txt, test_src.txt`.
We removed the duplicated documents in the KP20k training set according to the instructions in Rui Meng's Github. 
For each document, we sort all the present keyphrase labels according to their order of the first occurrence in the document. The absent keyphrase labels are then appended at the end of present keyphrase labels. 
Thanks to Mr. Wang Chen for his help on data preprocessing. 

For the training of our reinforced models, we use an additional token `<peos>` to mark the end of present keyphrases in the target files. 
The kp20k dataset with the `<peos>` token on the target files is located in the `kp20k_separated` directory. 

The `cross_domain_sorted` directory contains the test only datasets (inspec, krapivin, nus, and semeval). For example, the source and target file of nus dataset are `cross_domain_sorted/word_nus_testing_context.txt` and `cross_domain_sorted/word_nus_testing_allkeywords.txt`.


### Formats
* All the text should be tokenized and all the tokens should be separated by a space character.
* All digits should be replaced by a `<digit>` tokens.
* In source files, the title and the main body are separated by an `<eos>` token
* in target files, the keyphrases are separated by an `;` character. There is no space before and after the colon character, e.g., `keyphrase one;keyphrase two`. For the training of reinforced model, `<peos>` is used to mark the end of present ground-truth keyphrases, e.g., `present keyphrase one;present keyphrase two;<peos>;absent keyprhase one;absent keyphrase two`. 

## Training
### Train a baseline model
Please download and unzip the datasets in the `./data` directory.

1. Numericalize data. 

The `preprocess.py` script numericalizes the three pairs of source-target files, and produce the following files
`train.one2one.pt, train.one2many.pt, valid.one2one.pt, valid.one2many.pt, test.one2one.pt, test.one2many.pt, vocab.pt`.
The `*.one2one.pt` files which split a sample (source, {kp1, kp2, ...}) into multiple training sample (source, kp1), (source, kp2), ...
The `*.one2many.pt` files does not split the training sample. 

Command: `python3 preprocess.py -data_dir data/kp20k_sorted -remove_eos -include_peos`

To use the TG-Net model, you need to copy the directory `data/kp20k_sorted` to `data/kp20k_tg_sorted` and run the following preprocessing script. 
`python3 preprocess.py -data_dir data/kp20k_tg_sorted -remove_eos -include_peos -title_guided`

2. Train a baseline model using maximum-likelihood loss
  * catSeq: `python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -train_ml -one2many -one2many_mode 1 -batch_size 12 -seed 9527`
  * catSeqD: `python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -orthogonal_loss -lambda_orthogonal 0.03 -train_ml -one2many -one2many_mode 1 -use_target_encoder -batch_size 12 -seed 9527`
  * catSeqCorr: `python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -coverage_attn -review_attn -train_ml -one2many -one2many_mode 1 -batch_size 12 -seed 9527`
  * catSeqTG: `python3 train.py -data data/kp20k_tg_sorted/ -vocab data/kp20k_tg_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -title_guided -train_ml -one2many -one2many_mode 1 -batch_size 12 -batch_workers 3 -seed 9527`

### Train a reinforced model
Different from the baseline models, we use an additional token `<peos>` to mark the end of present keyphrases. See Section 3.2 of our paper. 

1. Numericalize data. 

Command: `python3 preprocess.py -data_dir data/kp20k_separated -remove_eos -include_peos`

To use the TG-Net model, you need to copy the directory `data/kp20k_separated` to `data/kp20k_tg_separated` and run the following preprocessing script. 
`python3 preprocess.py -data_dir data/kp20k_tg_separated -remove_eos -include_peos -title_guided`

2. Train ML
  * catSeq: `python3 train.py -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -train_ml -one2many -one2many_mode 1 -batch_size 12 -separate_present_absent -seed 9527`
  * catSeqD: `python3 train.py -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -orthogonal_loss -lambda_orthogonal 0.03 -train_ml -one2many -one2many_mode 1 -use_target_encoder -batch_size 12 -separate_present_absent -seed 9527`
  * catSeqCorr: `python3 train.py -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -coverage_attn -review_attn -train_ml -one2many -one2many_mode 1 -batch_size 12 -separate_present_absent -seed 9527`
  * catSeqTG: `python3 train.py -data data/kp20k_tg_separated/ -vocab data/kp20k_tg_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -title_guided -train_ml -one2many -one2many_mode 1 -batch_size 12 -batch_workers 3 -separate_present_absent -seed 9527`

3. Train RL
  * catSeq-2RF1: `python3 train.py -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -train_rl -one2many -one2many_mode 1 -batch_size 32 -separate_present_absent -pretrained_model [path_to_ml_pretrained_model] -max_length 60 -baseline self -reward_type 7 -replace_unk -topk G -seed 9527`
  * catSeqD-2RF1: `python3 train.py -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -use_target_encoder -train_rl -one2many -one2many_mode 1 -batch_size 32 -separate_present_absent -pretrained_model [path_to_ml_pretrained_model] -max_length 60 -baseline self -reward_type 7 -replace_unk -topk G -seed 9527`
  * catSeqCorr-2RF1: `python3 train.py -data data/kp20k_separated/ -vocab data/kp20k_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -coverage_attn -review_attn -train_rl -one2many -one2many_mode 1 -batch_size 32 -separate_present_absent -pretrained_model [path_to_ml_pretrained_model] -max_length 60 -baseline self -reward_type 7 -replace_unk -topk G -seed 9527`
  * catSeqTG-2RF1: `python3 train.py -data data/kp20k_tg_separated/ -vocab data/kp20k_tg_separated/ -exp_path exp/%s.%s -exp kp20k -epochs 20 -copy_attention -title_guided -train_rl -one2many -one2many_mode 1 -batch_size 32 -separate_present_absent -pretrained_model [path_to_ml_pretrained_model] -max_length 60 -baseline self -reward_type 7 -replace_unk -topk G -batch_workers 3 -seed 9527`


## Decode from a pretrained model
Following Yuan et al. 2018, we use greedy search to decode the keyphrases from a pre-trained model, but you increase the beam size by specifying the beam_size option.
  * catSeq on inspec dataset: `python3 interactive_predict.py -vocab data/kp20k_sorted/ -src_file data/cross_domain_sorted/word_inspec_testing_context.txt -pred_path pred/%s.%s -copy_attention -one2many -one2many_mode 1 -model [path_to_model] -max_length 60 -remove_title_eos -n_best 1 -max_eos_per_output_seq 1 -beam_size 1 -batch_size 20 -replace_unk`
  * catSeq-2RF1 on inspec dataset: `python3 interactive_predict.py -vocab data/kp20k_separated/ -src_file data/cross_domain_separated/word_inspec_testing_context.txt -pred_path pred/%s.%s -copy_attention -one2many -one2many_mode 1 -model [path_to_model] -max_length 60 -remove_title_eos -n_best 1 -max_eos_per_output_seq 1 -beam_size 1 -batch_size 20 -replace_unk -separate_present_absent`

For catseqCorr, and catseqCorr-2RF1, you need to add the options of `-coverage_attn -review_attn`. For catSeqD and catSeqD-2RF1, you need to add the options of `-use_target_encoder`. For catSeqTG, you need to add the options of `-title_guided` and change the vocab from `kp20k_sorted` to `kp20k_tg_sorted` (`kp20k_separated` to `kp20k_tg_separated`).
For other datasets, you need to change the option `-src_file` to the path of source file on other test dataset, but you do not need to change the `-vocab` option. 

Once the decoding finished, it creates a predictions.txt in the path specified by pred_path, e.g., pred/predict.kp20k.bi-directional.20180914-095220/predictions.txt. 
For each line in the prediction.txt contains all the predicted keyphrases for a source. 

## Compute evaluation score on prediction files
Command for computing the evaluation scores of a prediction file from a baseline model.

`python3 evaluate_prediction.py -pred_file_path [path_to_predictions.txt] -src_file_path [path_to_test_set_src_file] -trg_file_path [path_to_test_set_trg_file] -exp kp20k -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk -all_ks 5 M -present_ks 5 M -absent_ks 5 M`

Since the prediction files of reinforced models has a special token `<peos>`, we need to use the following command. 

`python3 evaluate_prediction.py -pred_file_path [path_to_predictions.txt] -src_file_path [path_to_test_set_src_file] -trg_file_path [path_to_test_set_trg_file] -exp kp20k -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk -all_ks 5 M -present_ks 5 M -absent_ks 5 M -prediction_separated`

## Enriched evaluation set
Please download the enriched trg file on the kp20k testing set from [here](https://drive.google.com/open?id=1ol8Flvc7RK84VbMG32m9ufSvoG6MMEYz) and extract it to `./data/kp20k_enriched`. We use the token `|` to separate name variations, e.g., `name variation 1 of keyphrase 1|name variation 2 of keyphrase 1;name variation 1 of keyphrase 2|name variation 2 of keyphrase 2`. 

Command for evaluating a baseline model:
`python3 evaluate_prediction.py -pred_file_path [path_to_predictions.txt] -src_file_path [path_to_kp20k_test_set_src_file] -trg_file_path data/kp20k_enriched/test_trg.txt -exp kp20k -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk -all_ks 5 M -present_ks 5 M -absent_ks 5 M -use_name_variations`

Command for evaluating a reinforced model:
`python3 evaluate_prediction.py -pred_file_path [path_to_predictions.txt] -src_file_path [path_to_kp20k_test_set_src_file] -trg_file_path data/kp20k_enriched/test_trg.txt -exp kp20k -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk -all_ks 5 M -present_ks 5 M -absent_ks 5 M -prediction_separated -use_name_variations`

## Test set output
The output files of our catSeqTG-2RF1 model are available [here](https://drive.google.com/open?id=1oTDwqhRE2uxX7Q0d9mtsRda_DI3-6DJZ). 

## Options
This section describe some common options for different python scripts. Please read the config.py for more details about the options. 

The options for the training script: 
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
-replace_unk: Replace the unk token with the token of highest attention score. 
```

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
-prediction_separated: the predictions has a special <peos> token. For the evaluation of the reinforced model. 
-use_name_variations: the target file contains name valuation set.
```

Some common options for rl training:
```
-train_rl: a flag for training a model using reward in a reinforcement learning setting.
-baseline []: specify the baseline for the policy gradient algorithm, choices=["none", "self"], "self" means we use self-critical as the baseline
-reward_type []: 0: f1, 1: recall, 2: ndcg, 3: accuracy, 4: alpha-ndcg, 5: alpha-dcg, 6: AP, 7: F1 (all duplicates are considered as incorrect)
-topk []: only pick the -topk predictions when computing the reward. M means use all the predictions to compute the reward. G is used to specify RF1 reward. If the number of predictions less than ground-truth, it will set k to the number of ground-truth keyphrases. Otherwise, it will set k to the number of predictions. The option `-reward_type 7 -topk G` yields the RF1 reward in our paper. 
-pretrained_model []: path of the MLE pre-trained model
-replace_unk: replace the unk token with the token that received the highest attention score.
-max_length []: max length of the output sequence
-num_predictions []: only effective when one2many_mode=2 or 3, control the number of predicted keyphrases.
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
We can also regularize the reward using the following two options. The baseline reward is not affected by the regularization. I tried it, but the performance is not good.
```
-regularization_type []: 0 means no regulaization, 1 means using percentage of unique keyphrases as regularization, 2 means using entropy of policy as regularization
-regularization_factor []: factor of regularization, regularized reward = (1-regularization_factor)*reward + regularization_factor*regularization
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
