import tensorflow as tf
import numpy as np
import glob, os, sys
import pickle
import math
import signal
import configparser
import ast
import argparse, re

from MACM_builder import GranuGateModel
from lib.tf_utils import non_neg_normalize
from lib.data_utils import list_shuffle, pad_batch_list
from lib.eval import write_run, compute_ndcg, compute_map
from lib.tf_utils import np_softmax, non_neg_normalize
from functools import partial


def load_dataset(path=None):
    '''load the train and test datasets'''
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def max_len(D):
    maxlen = 0
    for doc in D:
        current_len = len(doc)
        if current_len > maxlen:
            maxlen = current_len
    return maxlen

def prepare_data(data, block_size, max_q_len):
    """randomly sample a Q and then from its docs, randomly sample D+, D-,
    repeat this sampling until blocksize reached"""
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    while len(Q) < block_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_id = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_id = q_id[0]
        this_topic = q_list[q_id]
        if (1 not in data[this_topic]['query'] and len(data[this_topic]['query']) <= max_q_len):  # eliminate OOV
            query = data[this_topic]['query']
            docs = data[this_topic]['docs']
            scores = data[this_topic]['scores']
            if len(docs) >= 2:  # more than 2 docs in this group
                idx = np.random.choice(range(len(docs)), size=(2,), replace=False)
                if scores[idx[0]] != scores[idx[1]]:
                    Q.append(query)
                    if scores[idx[0]] > scores[idx[1]]:  # idx0 is pos doc
                        D_pos.append(docs[idx[0]])
                        D_neg.append(docs[idx[1]])
                        label.append([scores[idx[0]], scores[idx[1]]])
                    else:  # idx1 is pos doc
                        D_pos.append(docs[idx[1]])
                        D_neg.append(docs[idx[0]])
                        label.append([scores[idx[1]], scores[idx[0]]])
    return [Q, D_pos, D_neg, label]

def prepare_data_w_diff(data, block_size, max_q_len):
    "choose D+ D- with a BM25 score difference greater than a threshold"
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    while len(Q) < block_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_id = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_id = q_id[0]
        this_topic = q_list[q_id]
        if (1 not in data[this_topic]['query'] and len(data[this_topic]['query']) <= max_q_len):   # eliminate OOV
            query = data[this_topic]['query']
            docs = data[this_topic]['docs']
            scores = data[this_topic]['scores']
            if len(docs) >= 2:  # more than 2 docs in this group
                idx = np.random.choice(range(len(docs)), size=(2,), replace=False)
                if abs(scores[idx[0]] - scores[idx[1]]) >= 0.5:
                    Q.append(query)
                    if scores[idx[0]] > scores[idx[1]]:  # idx0 is pos doc
                        D_pos.append(docs[idx[0]])
                        D_neg.append(docs[idx[1]])
                        label.append([scores[idx[0]], scores[idx[1]]])
                    if scores[idx[1]] > scores[idx[0]]:  # idx1 is pos doc
                        D_pos.append(docs[idx[1]])
                        D_neg.append(docs[idx[0]])
                        label.append([scores[idx[1]], scores[idx[0]]])
    return [Q, D_pos, D_neg, label]

def prepare_data_all_Q(data, block_size, max_q_len):
    '''make use of all queries instead of randomly sampling queries
    for a given query, still sample docs uniformly
    (Q1, D1+, D1-)....(Q1, Dn+, Dn-), (Q2,D1+,D1-)...(Q2, Dn+, Dn-)
    block_size: for a given q, how many pairs of D+, D- generated
    '''
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            if len(docs) >=2:
                idx = np.random.choice(range(len(docs)), size=(block_size, 2), replace=True)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]

def prepare_data_sampleQ_BM25distro(data, q_sample_size, docpair_sample_size, max_q_len):
    """sample randomly a query
    for a given query, sample doc according to the distro softmax(BM25_scores)
    q_sample_size: num of queries sampled from one data pkl file
    docpair_sample_size: for each q, how many pairs of (D+, D-) sampled
    """
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    Q_counter = 0
    while Q_counter < q_sample_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_idx = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_idx = q_idx[0]  # idx from 0 to len(q_list)
        topic_num = q_list[q_idx]
        query = data[topic_num]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[topic_num]['docs']
            scores = data[topic_num]['scores']
            if len(docs) >=2:
                # calcuate BM25 score softmax distribution
                np_scores = np.asarray(scores)
                BM25_distro = np_softmax(np_scores)
                idx = np.random.choice(range(len(docs)), size=(docpair_sample_size, 2), replace=True, p=BM25_distro)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
                Q_counter += 1
    return [Q, D_pos, D_neg, label]

def prepare_data_all_Q_BM25distro(data, block_size, max_q_len):
    """make use of all queries instead of randomly sampling queries
    for a given query, sample doc according to the distro softmax(BM25_scores)
    block_size: for a given q, how many pairs of D+, D- sampled
    """
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            if len(docs) >=2:
                # calcuate BM25 score softmax distribution
                np_scores = np.asarray(scores)
                BM25_distro = np_softmax(np_scores)
                idx = np.random.choice(range(len(docs)), size=(block_size, 2), replace=True, p=BM25_distro)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]


def prepare_data_allQ_allD(data, max_q_len):
    '''make use of all queries and all docs
    for a given query q, list all possible document pair combinations, and assign to D_pos, D_neg
    according to its score
    '''
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            if len(docs) >=2:
                idx = []
                idx.extend(combinations(range(len(docs)), 2))
                for i in range(len(idx)):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]

def int_handler(sess, model_path, saver, signal, frame):
    '''ctrl+C interrupt handler'''
    print('You pressed Ctrl+C!, model will be saved and stopping training now')
    save_path = saver.save(sess, model_path)
    print('successfully saved model to {}'.format(model_path))
    sys.exit(0)

def train(config_path, resume_training=False):
    '''training process'''
    # parse config
    config = configparser.ConfigParser()
    config.read(config_path)
    '''hyper params'''
    model_name_str = config['hyperparams']['model_name_str']
    batch_size = ast.literal_eval(config['hyperparams']['batch_size'])  # batch_size
    vocab_size = ast.literal_eval(config['hyperparams']['vocab_size'])  # vocab_size
    emb_dim = ast.literal_eval(config['hyperparams']['emb_dim'])  # embedding dimension
    n_gramsets = ast.literal_eval(config['hyperparams']['n_gramsets'])
    n_filters1 = ast.literal_eval(config['hyperparams']['n_filters1'])
    n_filters2 = ast.literal_eval(config['hyperparams']['n_filters2'])
    kernel_sizes1 = ast.literal_eval(config['hyperparams']['kernel_sizes1'])
    kernel_sizes2 = ast.literal_eval(config['hyperparams']['kernel_sizes2'])
    conv_strides1 = ast.literal_eval(config['hyperparams']['conv_strides1'])
    conv_strides2 = ast.literal_eval(config['hyperparams']['conv_strides2'])
    pool_sizes0 = ast.literal_eval(config['hyperparams']['pool_sizes0'])
    pool_sizes1 = ast.literal_eval(config['hyperparams']['pool_sizes1'])
    pool_sizes2 = ast.literal_eval(config['hyperparams']['pool_sizes2'])
    pool_strides = ast.literal_eval(config['hyperparams']['pool_strides'])
    n_hidden_layers = ast.literal_eval(config['hyperparams']['n_hidden_layers'])  # num hidden layers
    hidden_sizes = ast.literal_eval(config['hyperparams']['hidden_sizes'])
    hinge_margin = ast.literal_eval(config['hyperparams']['hinge_margin'])
    train_datablock_size = ast.literal_eval(config['hyperparams']['train_datablock_size'])
    # for sampleQ sampleD prepare_data()
    q_sample_size = ast.literal_eval(config['hyperparams']['q_sample_size'])
    docpair_sample_size = ast.literal_eval(config['hyperparams']['docpair_sample_size'])
    n_epoch = ast.literal_eval(config['hyperparams']['n_epoch']) # num of epochs
    alpha = ast.literal_eval(config['hyperparams']['alpha'])  # weight decay
    # q and doc cuts
    q_len = ast.literal_eval(config['hyperparams']['q_len'])
    d_len = ast.literal_eval(config['hyperparams']['d_len'])
    # model path and datapath
    model_base_path = config['hyperparams']['model_base_path']
    data_base_path = config['hyperparams']['data_base_path']
    
    '''TRAINING DIR'''
    TRAIN_DIR = '{}/train/'.format(data_base_path)
    train_files = glob.glob("{}/data*.pkl".format(TRAIN_DIR))

    '''build model'''
    model = GranuGateModel(
    vocab_size=vocab_size, emb_dim=emb_dim, n_gramsets=n_gramsets, n_filters1=n_filters1,
                 n_filters2=n_filters2,
                 kernel_sizes1=kernel_sizes1, kernel_sizes2=kernel_sizes2,
                 conv_strides1=conv_strides1, conv_strides2=conv_strides2,
                 pool_sizes0=pool_sizes0, pool_sizes1=pool_sizes1,
                 pool_sizes2=pool_sizes2,
                 pool_strides=pool_strides,
                 n_mlp_layers=n_hidden_layers,
                 hidden_sizes=hidden_sizes
    )
    '''init with pretrained embeddings'''
    model.load_emb(emb_path='{}/glove_emb.pkl'.format(data_base_path))
    # placeholders
    x_q = tf.placeholder(tf.int32, [None, q_len], name='x_q')
    x_d_pos = tf.placeholder(tf.int32, [None, d_len], name='x_d_pos')
    x_d_neg = tf.placeholder(tf.int32, [None, d_len], name='x_d_neg')
    l = tf.placeholder(tf.float32, [None, 2], name='l')
    keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')  # scalar placeholder to control application of dropout in training and testing
    # cost
    R_pos, R_neg = model.apply_rel(x_q, x_d_pos, x_d_neg, keep_prob)  # shape=(batchsize, )
    cost = tf.reduce_mean((hinge_margin - tf.sign(l[:, 0] - l[:, 1]) * (R_pos - R_neg)) *
    tf.cast((hinge_margin - tf.sign(l[:, 0] - l[:, 1]) * (R_pos - R_neg)) > 0, tf.float32))  # weight decay already applied in layer modules
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(cost)
    # init variables
    init = tf.global_variables_initializer()
    # saver to save model
    saver = tf.train.Saver()
    # experiment
    print("Experiment")
    if resume_training == False:
        f_log = open('{}/{}/logs/training_log.txt'.format(model_base_path, model_name_str), 'w+', 1)
        valid_log = open('{}/{}/logs/valid_log.txt'.format(model_base_path, model_name_str), 'w+', 1)
    else:
        f_log = open('{}/{}/logs/training_log.txt'.format(model_base_path, model_name_str), 'a+', 1)
        valid_log = open('{}/{}/logs/valid_log.txt'.format(model_base_path, model_name_str), 'a+', 1)
    # model_file
    model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement=True
    with tf.Session(config=config) as sess:
        # listen interrupt signal, pass model path to int_handler to save model
        # signal.signal(signal.SIGINT, partial(int_handler, sess, model_file, saver))
        # run init variables
        sess.run(init)
        # init best validation MAP value
        best_MAP = 0.0
        # restore saved parameter if resume_training is true
        if resume_training == True:
            model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
            saver.restore(sess, model_file)
            with open('{}/{}/saves/best_MAP.pkl'.format(model_base_path, model_name_str), 'rb') as f_MAP:
                best_MAP = pickle.load(f_MAP)
            print("loaded model, and resume training now")
        # preparing batch data
        file_count = 0
        for epoch in range(1, n_epoch + 1):
            train_files = list_shuffle(train_files)
            for f in train_files:
                '''load_data'''
                data = load_dataset(f)
                print("loaded {}".format(f))
                '''prepare_data'''
                [Q, D_pos, D_neg, L] = prepare_data_sampleQ_BM25distro(data, q_sample_size, docpair_sample_size, q_len)
                ''' shuffle data'''
                train_data = list_shuffle(Q, D_pos, D_neg, L)
                '''training func'''
                batch_count_tr = 0
                num_batch = len(train_data[0]) // batch_size
                for batch_count in range(num_batch):
                    Q = train_data[0][batch_size * batch_count: batch_size * (batch_count + 1)]
                    D_pos = train_data[1][batch_size * batch_count: batch_size * (batch_count + 1)]
                    D_neg = train_data[2][batch_size * batch_count: batch_size * (batch_count + 1)]
                    L = train_data[3][batch_size * batch_count: batch_size * (batch_count + 1)]
                    Q = np.asarray(pad_batch_list(Q, max_len=q_len, padding_id=0), dtype=np.int32)
                    D_pos = np.asarray(pad_batch_list(D_pos, max_len=d_len, padding_id=0), dtype=np.int32)
                    D_neg = np.asarray(pad_batch_list(D_neg, max_len=d_len, padding_id=0), dtype=np.int32)
                    L = np.asarray(L, dtype=np.float32)
                    # run optimizer on this batch
                    sess.run(train_op, feed_dict={x_q: Q, x_d_pos: D_pos, x_d_neg: D_neg, l: L, keep_prob: 1.0})
                    cost_value = sess.run(cost, feed_dict={x_q: Q, x_d_pos: D_pos, x_d_neg: D_neg, l: L, keep_prob: 1.0})
                    batch_count_tr +=1
                    print("epoch {} batch {} training cost: {}, param_norm: , grad_norm: , w_weight_norm: , w_grad_norm: " \
                    .format(epoch, batch_count_tr, cost_value))
                    f_log.write("epoch {} batch {} training cost: {}".format(epoch, batch_count_tr, cost_value) + '\n')
                file_count += 1
                if file_count % 4 == 0:
                    # do rapid validation
                    ndcg_list, mapvalue = validation(sess,
                    [x_q, x_d_pos, R_pos, keep_prob], data_base_path, model_base_path,
                    model_name_str, dataset, d_len)  # pass the training compuational graph placeholder to valid function to evaluate with the same set of parameters
                    print("epoch :{}, pkl count: {}, NDCG".format(epoch, file_count), ndcg_list)
                    print("MAP: {}".format(mapvalue))
                    # check if this valid period is the best and update best_MAP, save model to disk
                    if mapvalue > best_MAP:
                        best_MAP = mapvalue
                        with open('{}/{}/saves/best_MAP.pkl'.format(model_base_path, model_name_str), 'wb') as f_MAP:
                            pickle.dump(best_MAP, f_MAP)
                        # save model params after several epoch
                        model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
                        save_path = saver.save(sess, model_file)
                        print("successfully saved model to the path {}".format(save_path))
                    valid_log.write("{} {} {} {}".format(ndcg_list[1][0], ndcg_list[1][1], ndcg_list[1][2], ndcg_list[1][3]))
                    valid_log.write(" MAP: {}".format(mapvalue))
                    valid_log.write('\n')
        f_log.close()
        valid_log.close()

def validation(sess, placeholder_list, data_base_path, model_base_path,
               model_name_str, dataset, d_len, compute_ndcg_flag=True):
    '''modelpath: model path
    placeholder_list: placeholder list of [x_q, x_d, R, keep_prob] from the outer function
    '''
    '''hyper params'''
    batch_size = 128  # batch_size
    # q and doc cuts
    q_len = 15
    d_len = d_len
    '''VALID DIR'''
    if dataset == "ClueWeb":
        TEST_DIR = '{}/valid/WT0912/'.format(data_base_path)
        RESULTS_DIR = '{}/{}/result/valid/'.format(model_base_path, model_name_str)
        test_files = glob.glob("{}/data*.pkl".format(TEST_DIR))
    if dataset == "Robust":
        TEST_DIR = '{}/valid/'.format(data_base_path)
        RESULTS_DIR = '{}/{}/result/valid/'.format(model_base_path, model_name_str)
        test_files = glob.glob("{}/data*.pkl".format(TEST_DIR))

    '''build model'''
    # model = pointmodel(vocab_size=vocab_size, emb_dim=emb_dim, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
    # placeholders
    x_q = placeholder_list[0]
    x_d = placeholder_list[1]
    R = placeholder_list[2]
    keep_prob = placeholder_list[3]
    # run path
    run_path = RESULTS_DIR + 'run.txt'
    # run list containing run lines of all topics
    all_run_list = []
    # load one testfile at a time and conduct test
    for f in test_files:
        # list to contains scores
        # load testdata data = {'topic_num': {'query':[], 'docs':[], 'docno':[]}}
        data = load_dataset(f)
        print("len valid data", len(data))
        # generate full format [Q, D, meta_dict] meta_dict={'topic_num':[], 'docno':[]} for one topic group
        for topic_num in data:
            Q = []
            D = []
            meta_dict = {'topic_num':[], 'docno':[]}
            batch_id = 0
            num_batch = int(math.ceil(len(data[topic_num]['docs']) * 1.0 / batch_size))
            for i in range(len(data[topic_num]['docs'])):
                Q.append(data[topic_num]['query'])
                D.append(data[topic_num]['docs'][i])
                meta_dict['topic_num'].append(topic_num)
                meta_dict['docno'].append(data[topic_num]['docno'][i])
            # padding
            Q_test = np.asarray(pad_batch_list(Q, max_len=q_len, padding_id=0), dtype=np.int32)
            D_test = np.asarray(pad_batch_list(D, max_len=d_len, padding_id=0), dtype=np.int32)
            scores = []
            for batch_id in range(num_batch):
                Q_value = Q_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                D_value = D_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                batch_rel = sess.run(R, feed_dict={x_q: Q_value, x_d: D_value, keep_prob: 1.0})  # in test phase, no dropout
                batch_scores = batch_rel.tolist()
                scores += batch_scores
            np_scores = np.asarray(scores)
            np_scores = non_neg_normalize(np_scores)
            scores = np_scores.tolist()
            run_list = zip(meta_dict['topic_num'], meta_dict['docno'], scores)
            print("run_file for topic {} created".format(topic_num))
            all_run_list += run_list
    write_run(all_run_list, run_path)

    if compute_ndcg_flag == True:
        # prepare run file list
        rel_path = '{}/{}/tmp/valid/qrels.1-200.clueweb'.format(model_base_path, model_name_str)
        tmp_path = '{}/{}/tmp/valid/temp.txt'.format(model_base_path, model_name_str)
        # compute ndcg by calling external tools
        ndcg_list = compute_ndcg(run_path, rel_path, tmp_path)
        mapvalue = compute_map(run_path, rel_path, tmp_path)
        return ndcg_list, mapvalue

def test(model_path, config_path, compute_ndcg_flag=True):
    # parse config
    config = configparser.ConfigParser()
    config.read(config_path)
    '''hyper params'''
    model_name_str = config['hyperparams']['model_name_str']
    batch_size = ast.literal_eval(config['hyperparams']['batch_size'])  # batch_size
    vocab_size = ast.literal_eval(config['hyperparams']['vocab_size'])  # vocab_size
    emb_dim = ast.literal_eval(config['hyperparams']['emb_dim'])  # embedding dimension
    n_gramsets = ast.literal_eval(config['hyperparams']['n_gramsets'])
    n_filters1 = ast.literal_eval(config['hyperparams']['n_filters1'])
    n_filters2 = ast.literal_eval(config['hyperparams']['n_filters2'])
    kernel_sizes1 = ast.literal_eval(config['hyperparams']['kernel_sizes1'])
    kernel_sizes2 = ast.literal_eval(config['hyperparams']['kernel_sizes2'])
    conv_strides1 = ast.literal_eval(config['hyperparams']['conv_strides1'])
    conv_strides2 = ast.literal_eval(config['hyperparams']['conv_strides2'])
    pool_sizes0 = ast.literal_eval(config['hyperparams']['pool_sizes0'])
    pool_sizes1 = ast.literal_eval(config['hyperparams']['pool_sizes1'])
    pool_sizes2 = ast.literal_eval(config['hyperparams']['pool_sizes2'])
    pool_strides = ast.literal_eval(config['hyperparams']['pool_strides'])
    n_hidden_layers = ast.literal_eval(config['hyperparams']['n_hidden_layers'])  # num hidden layers
    hidden_sizes = ast.literal_eval(config['hyperparams']['hidden_sizes'])
    hinge_margin = ast.literal_eval(config['hyperparams']['hinge_margin'])
    # q and doc cuts
    q_len = ast.literal_eval(config['hyperparams']['q_len'])
    d_len = ast.literal_eval(config['hyperparams']['d_len'])
    # model and data base paths
    data_base_path = config['hyperparams']['data_base_path']
    model_base_path = config['hyperparams']['model_base_path']
    # dataset
    dataset = config['hyperparams']['dataset']

    '''TEST DIR'''
    if dataset == "ClueWeb":
        TEST_DIR = '{}/test51-200/WT0912/'.format(data_base_path)
        RESULTS_DIR = '{}/{}/result/test/'.format(model_base_path, model_name_str)
        test_files = glob.glob("{}/data*.pkl".format(TEST_DIR))
    if dataset == "Robust":
        TEST_DIR = '{}/test/'.format(data_base_path)
        RESULTS_DIR = '{}/{}/result/test/'.format(model_base_path, model_name_str)
        test_files = glob.glob("{}/data*.pkl".format(TEST_DIR))
    '''build model'''

    '''build model'''
    model = GranuGateModel(
    vocab_size=vocab_size, emb_dim=emb_dim, n_gramsets=n_gramsets, n_filters1=n_filters1,
                 n_filters2=n_filters2,
                 kernel_sizes1=kernel_sizes1, kernel_sizes2=kernel_sizes2,
                 conv_strides1=conv_strides1, conv_strides2=conv_strides2,
                 pool_sizes0=pool_sizes0, pool_sizes1=pool_sizes1,
                 pool_sizes2=pool_sizes2,
                 pool_strides=pool_strides,
                 n_mlp_layers=n_hidden_layers,
                 hidden_sizes=hidden_sizes
    )
    '''init with pretrained embeddings to add emb_mat var'''
    model.load_emb(emb_path='/data/rali5/Tmp/nieyifan/aol/glove_emb.pkl')
    # placeholders
    x_q = tf.placeholder(tf.int32, [None, q_len])
    x_d_pos = tf.placeholder(tf.int32, [None, d_len])
    x_d_neg = tf.placeholder(tf.int32, [None, d_len])
    l = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')
    # cost
    R_pos, R_neg = model.apply_rel(x_q, x_d_pos, x_d_neg, keep_prob)
    cost = tf.reduce_mean((hinge_margin - tf.sign(l[:, 0] - l[:, 1]) * (R_pos - R_neg)) *
    tf.cast((hinge_margin - tf.sign(l[:, 0] - l[:, 1]) * (R_pos - R_neg)) > 0, tf.float32) )
    # optimizer
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.999)
    #train_op = optimizer.minimize(cost)
    # init variables
    init = tf.global_variables_initializer()
    # saver to save model
    saver = tf.train.Saver()
    # validation
    with tf.Session() as sess:
        # load model from file
        saver.restore(sess, model_path)
        print("loaded model, and perform test now")
        # define output run file path
        run_path = RESULTS_DIR + 'run.txt'
        # run list containing run lines of all topics
        all_run_list = []
        # load one testfile at a time and conduct test
        for f in test_files:
            # list to contains scores
            # load testdata data = {'topic_num': {'query':[], 'docs':[], 'docno':[]}}
            data = load_dataset(f)
            # generate full format [Q, D, meta_dict] meta_dict={'topic_num':[], 'docno':[]} for one topic group
            for topic_num in data:
                Q = []
                D = []
                meta_dict = {'topic_num':[], 'docno':[]}
                batch_id = 0
                num_batch = int(math.ceil(len(data[topic_num]['docs']) * 1.0 / batch_size))
                for i in range(len(data[topic_num]['docs'])):
                    Q.append(data[topic_num]['query'])
                    D.append(data[topic_num]['docs'][i])
                    meta_dict['topic_num'].append(topic_num)
                    meta_dict['docno'].append(data[topic_num]['docno'][i])
                # padding
                Q_test = np.asarray(pad_batch_list(Q, max_len=q_len, padding_id=0), dtype=np.int32)
                D_test = np.asarray(pad_batch_list(D, max_len=d_len, padding_id=0), dtype=np.int32)
                scores = []
                for batch_id in range(num_batch):
                    Q_value = Q_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                    D_value = D_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                    batch_rel = sess.run(R_pos, feed_dict={x_q: Q_value, x_d_pos: D_value, x_d_neg: D_value, keep_prob: 1.0})
                    batch_scores = batch_rel.tolist()
                    scores += batch_scores
                np_scores = np.asarray(scores)
                np_scores = non_neg_normalize(np_scores)
                scores = np_scores.tolist()
                run_list = zip(meta_dict['topic_num'], meta_dict['docno'], scores)
                print("run_file for topic {} created".format(topic_num))
                all_run_list += run_list
        write_run(all_run_list, run_path)

        if compute_ndcg_flag==True:
            rel_path = '{}/{}/tmp/test/qrels.1-200.clueweb'.format(model_base_path, model_name_str)
            tmp_path = '{}/{}/tmp/test/temp.txt'.format(model_base_path, model_name_str)
            # compute ndcg by calling external tools
            ndcg_list = compute_ndcg(run_path, rel_path, tmp_path)
            mapvalue = compute_map(run_path, rel_path, tmp_path)
            return ndcg_list, mapvalue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--resume", type=str, default="True")
    args = parser.parse_args()

    if args.mode == "train":
        if args.resume == "True":
            train(args.path, resume_training=True)
        elif args.resume == "False":
            train(args.path, resume_training=False)
        else:
            raise ValueError("invalid resume flag")
    if args.mode == "test":
        config = configparser.ConfigParser()
        config.read(args.path)
        model_name_str = config['hyperparams']['model_name_str']
        model_base_path = config['hyperparams']['model_base_path']
        ndcg_list, mapvalue = test('{}/{}/saves/model_file'.format(model_base_path, model_name_str), config_path)
        print(ndcg_list, mapvalue)

if __name__ == '__main__':
    main()
