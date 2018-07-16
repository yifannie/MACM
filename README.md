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
'''
qid should be a str type, e.g. '51'

