# MACM

This in the implementation of the Multi-level Abstraction Convolutional Model (MACM) in the paper [Multi-level Abstraction Convolutional Model for Information Retrieval with Weak Supervision](https://dl.acm.org/citation.cfm?id=3210123)

if you use this code for your work, please cite it as

```
Yifan Nie, Alessandro Sordoni, Jian-Yun Nie:
Multi-level Abstraction Convolutional Model with Weak Supervision for Information Retrieval. SIGIR 2018: 985-988
```
bibtex
```
@inproceedings{DBLP:conf/sigir/NieSN18,
author    = {Yifan Nie and
Alessandro Sordoni and
Jian{-}Yun Nie},
title     = {Multi-level Abstraction Convolutional Model with Weak Supervision
for Information Retrieval},
booktitle = {The 41st International {ACM} {SIGIR} Conference on Research {\&}
Development in Information Retrieval, {SIGIR} 2018, Ann Arbor, MI,
USA, July 08-12, 2018},
pages     = {985--988},
year      = {2018}
}
```

## prerequisites

* Python 3.6.3
* Tensorflow 1.3.1
* Numpy

## Usage

### Congigure: first, configure the hyperparameter through the config file, a sample is provided

[sample.config](https://github.com/yifannie/MACM/blob/master/sample.config)

A fold for saving models should be created on local disk and should contain the config file and 4 sub-folders:

```
model
..\config
..\logs
..\result
..\saves
..\tmp

```

### Train

To train the model, pass the config file path, mode, and resume flag into command line
```
python macm_train.py --path: path to the config file \
--mode train  (train or test) \
--resume False  (whether to resume training from the saved model or train for a brand new model)
```

### Test
To test the model, pass the config file path, mode into command line
```
python macm_train.py --path: path to the config file \
--mode test  (train or test) 
```

## Data Preprocessing
All queries and documents should be encoded into sequences of integer term ids, term id should begin with 1, where 1 indicates OOV term.
Training data should be stored in python dict with the following structure:
```
data = {
   qid:{'query': [257, 86, 114],
        'docs': [[123, 456, 6784...], [235, 345, 768,...],...]
        'scores': [25.16, 16.83, ...]
   }
}
```
qid should be a str type, e.g. '31'

Validation or testing data should be stored in python dict with the following structure:
```
test_data = {
  qid:{'query': [257, 86, 114],
       'docs': [[123, 456, 6784...], [235, 345, 768,...],...]
       'docno': ['clueweb09-en0000-00-00000', 'clueweb09-en0000-00-00001'...]
  }
}
``` 
qid should be a str type, e.g. '51'

A set of pre-trained embeddings should also be specified to input into the model. In this paper we employed [GloVe6B.300d](https://nlp.stanford.edu/projects/glove/), you can also train models like word2vec on your own corpus. The embedding data should be a pickled file containging an embedding matrix of shape [vocab_size, emb_dim] of data type float32, and stored under the base_data_path (configured in the config file).
## Configurations

* model_name_str: model folder's name
* batch_size: batchsize
* vocab_size: vocabulary size
* emb_size: embedding size
* n_gramsets: num of parallel convolutions with different filter size for each conv layer, should be set to 1 for the MACM model
* n_filters1: num of filters for the first conv layer, should be in brackets, e.g. [32]
* n_filters2: num of filters for the second conv layer, should be in brackets, e.g. [16]
* kernel_sizes1: filter shape of conv layer 1
* kernel_sizes2: filter shape of conv layer 2
* conv_strides: stride for all the conv layers
* pool_sizes0: pooling size for the interaction matrix
* pool_sizes1: pooling size for the 1st conv layer
* pool_sizes2: pooling size for the 2nd conv layer
* pool_strides: stride for all pooling operactions
* n_hiddden_layers: num of hidden layers in the flattened MLP after each conv layer
* hidden_sizes: hidden size for mlp hidden layers
* hinge_margin: hinge margin for pairwise training loss
* train_datablock_size: training data block size to store a block of training examples in RAM during training
* q_sample_size: training query sampling size if using sampling to sample a subset of training queries
* docpair_sample_size: document pair sampling size if using sampling to sample a subset of training samples
* n_epoch: max num of epochs
* alpha: L2 normalization weight
* q_len: max query length (num of query terms)
* d_len: max document length (num of document terms)
* model_base_path: full path to the parent folder of the model folder (i.e. if model folder is located in /scratch/models/macm, this base path should be /scratch/models/)
* data_base_path: full path to the parent folder of the training data folder (i.e. if the train data folder is located in /scratch/data/train/, this base path should be /scratch/data/)


