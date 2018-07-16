import tensorflow as tf
import numpy as np
import pickle

from lib.tf_utils import masked_softmax, batch_cossim
from lib.tf_utils import MaxM_fromBatch, MaxM_from4D

class GranuGateModel(object):
    '''implements a score model of one layer of'''
    def __init__(self, vocab_size, emb_dim=300, n_gramsets=1, n_filters1=[10],
                 n_filters2=[10],
                 kernel_sizes1=[(3, 3)], kernel_sizes2=[(3, 3)],
                 conv_strides1=[(1, 1)], conv_strides2=[(1, 1)],
                 pool_sizes0=[(2, 2)],
                 pool_sizes1=[(2, 2)], pool_sizes2=[(2, 2)],
                 pool_strides=[(1, 1)], n_mlp_layers=1,
                 hidden_sizes=[256]):
        '''vocab_size: vocab_size
        emb_dim: embedding dimension
        n_gramsets: num of different parallel conv layers for different windowsize
        n_filters: a list of ints for num of filters for each layer len(n_filters) == n_gramsets
        kernel_sizes: a list of tuples for each conv layer's kernel_size len(kernel_sizes) == n_gramsets
        conv_strides: a list of tuples for each conv layer's strides
        pool_strides: a list of tuples for each pooling layer's strides
        n_mlp_layer: num of final MLP hidden layers
        hidden_sizes: a list of ints of hidden size of each of the MLP hidden layers
        '''
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_gramsets = n_gramsets
        if len(n_filters1) != self.n_gramsets:
            raise ValueError("n_filters list len not eq to n_gramsets")
        self.n_filters1 = n_filters1
        if len(n_filters2) != self.n_gramsets:
            raise ValueError("n_filters list len not eq to n_gramsets")
        self.n_filters2 = n_filters2
        if len(kernel_sizes1) != self.n_gramsets:
            raise ValueError("kernel_size list len not eq to n_gramsets")
        self.kernel_sizes1 = kernel_sizes1
        if len(kernel_sizes2) != self.n_gramsets:
            raise ValueError("kernel_size list len not eq to n_gramsets")
        self.kernel_sizes2 = kernel_sizes2
        if len(conv_strides1) != self.n_gramsets:
            raise ValueError("conv_strides list len not eq to n_gramsets")
        self.conv_strides1 = conv_strides1
        if len(conv_strides2) != self.n_gramsets:
            raise ValueError("conv_strides list len not eq to n_gramsets")
        self.conv_strides2 = conv_strides2
        if len(pool_sizes1) != self.n_gramsets:
            raise ValueError("pool_sizes list len not eq to n_gramsets")
        self.pool_sizes1 = pool_sizes1
        if len(pool_sizes2) != self.n_gramsets:
            raise ValueError("pool_sizes list len not eq to n_gramsets")
        self.pool_sizes2 = pool_sizes2

        self.pool_sizes0 = pool_sizes0
        if len(pool_strides) != self.n_gramsets:
            raise ValueError("pool_strides list len not eq to n_gramsets")
        self.pool_strides = pool_strides
        self.n_mlp_layers = n_mlp_layers
        if len(hidden_sizes) != self.n_mlp_layers:
            raise ValueError("hidden_sizes list len not eq to n_gramsets")
        self.hidden_sizes = hidden_sizes
        '''embedding matrix'''
        #self.emb_mat = tf.get_variable("emb_mat", shape=[vocab_size, emb_dim], dtype=tf.float32, initializer=tf.orthogonal_initializer(1.0))
        self.gate = tf.get_variable("gate", shape=[3,])

    def load_emb(self, emb_path):
        with open(emb_path, 'rb') as f:
            np_emb = pickle.load(f, encoding="latin1")
        print(np_emb.shape)
        emb_init = tf.constant_initializer(np_emb)
        "Embedding Matrix"
        self.emb_mat = tf.get_variable("emb_mat", shape=[self.vocab_size, self.emb_dim],
         dtype=tf.float32, initializer=emb_init, trainable = False)
        print("successfully initialized the emb_mat")

    def apply_rel(self, x_q, x_d_pos, x_d_neg, keep_prob):
        '''apply rel
        x_q: query matrix (batch_size, q_len)
        x_d_pos: document of shape (batchsize, d_len)
        x_d_neg: document of shape (batchsize, d_len)
        l: BM25 training label
        returns: L cost
        '''
        # define weight regularizer
        L2_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        # prepare input matrix
        q_emb = tf.gather(self.emb_mat, x_q)  # (BS, q_len, emb_dim)
        d_emb_pos = tf.gather(self.emb_mat, x_d_pos)  # (BS, d_len, emb_dim)
        d_emb_neg = tf.gather(self.emb_mat, x_d_neg)  # (BS, d_len, emb_dim)
        interact_mat1_pos = batch_cossim(q_emb, d_emb_pos)  # (BS, q_len, d_len)
        interact_mat1_neg = batch_cossim(q_emb, d_emb_neg)  # (BS, q_len, d_len)
        input_mat1_pos = tf.expand_dims(interact_mat1_pos, axis=3)  # (BS, qlen, d_len, 1) = (BS, n_rows, n_cols, n_chans)
        input_mat1_neg = tf.expand_dims(interact_mat1_neg, axis=3)  # (BS, qlen, d_len, 1) = (BS, n_rows, n_cols, n_chans)
        M1_pos = MaxM_fromBatch(interact_mat1_pos)  # (BS,)
        M1_neg = MaxM_fromBatch(interact_mat1_neg)  # (BS,)
        ## level1 convs layers
        level1_pooled_h_pos = []  # to store conved and pooled hiddens for differnt windowsize sets
        level1_pooled_h_neg = []  # equi to level2 interact mat
        M2_pos_L = []
        M2_neg_L = []
        for i in range(self.n_gramsets):  # total num of conv layers
            scope_name = "conv_level1_set{}".format(i)
            with tf.variable_scope(scope_name):
                # pos part
                conv_pos = tf.layers.conv2d(
                        inputs=input_mat1_pos,  # (BS, q_len, d_len, 1) channel_last
                        filters=self.n_filters1[i],
                        kernel_size=self.kernel_sizes1[i],
                        padding='valid',
                        strides=self.conv_strides1[i],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                    uniform=True, seed=None, dtype=tf.float32),
                        kernel_regularizer=L2_regularizer,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        bias_regularizer=L2_regularizer,
                        name=None,
                        reuse=False)
                pool_pos = tf.layers.max_pooling2d(  # shape=(BS, q_len, d_len, 1)
                        inputs=conv_pos,
                        pool_size=self.pool_sizes1[i],
                        strides=self.pool_strides[i],
                        name=None)
                #pool_pos = pool_pos[:, :, :, 0]  # shape=(BS, q_len, d_len) drop last chan axe
                level1_pooled_h_pos.append(pool_pos)
                # neg part
                conv_neg = tf.layers.conv2d(
                        inputs=input_mat1_neg,  # (BS, q_len, d_len, 1) channel_last
                        filters=self.n_filters1[i],
                        kernel_size=self.kernel_sizes1[i],
                        padding='valid',
                        strides=self.conv_strides1[i],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                    uniform=True, seed=None, dtype=tf.float32),
                        kernel_regularizer=L2_regularizer,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        bias_regularizer=L2_regularizer,
                        name=None,
                        reuse=True)
                pool_neg = tf.layers.max_pooling2d(  # shape=(BS, q_len, d_len, 1)
                        inputs=conv_neg,
                        pool_size=self.pool_sizes1[i],
                        strides=self.pool_strides[i],
                        name=None)
                #pool_neg = pool_neg[:, :, :, 0]  # make shape (BS, q_len, d_len) ,drop channel axis
                level1_pooled_h_neg.append(pool_neg)
                M2_pos_L.append(MaxM_from4D(conv_pos))  # ngramsets * (BS, )
                M2_neg_L.append(MaxM_from4D(conv_neg))  # ngramsets * (BS, )
        M2_pos = tf.stack(M2_pos_L, axis=1)  # (BS, ngramsets)
        M2_neg = tf.stack(M2_neg_L, axis=1)  # (BS, ngramsets)
        M2_pos = tf.reduce_mean(M2_pos, axis=1)  # (BS,)
        M2_neg = tf.reduce_mean(M2_neg, axis=1)  # (BS,)
        ### level2 conv layers
        level2_pooled_h_pos = []  # to store conved and pooled hiddens of different windowsizes
        level2_pooled_h_neg = []  # equi to level3 interact_mat
        M3_pos_L = []
        M3_neg_L = []
        for i in range(self.n_gramsets):  # total num of sets of windowsizes
            scope_name = "conv_level2_set_{}".format(i)
            with tf.variable_scope(scope_name):
                # pos part
                conv2_pos = tf.layers.conv2d(
                        inputs=level1_pooled_h_pos[i],  # (BS, q_len, d_len, 1) channel_last
                        filters=self.n_filters2[i],
                        kernel_size=self.kernel_sizes2[i],
                        padding='valid',
                        strides=self.conv_strides2[i],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                    uniform=True, seed=None, dtype=tf.float32),
                        kernel_regularizer=L2_regularizer,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        bias_regularizer=L2_regularizer,
                        name=None,
                        reuse=False)
                pool2_pos = tf.layers.max_pooling2d(
                        inputs=conv2_pos,
                        pool_size=self.pool_sizes2[i],
                        strides=self.pool_strides[i],
                        name=None)
                level2_pooled_h_pos.append(pool2_pos)
                # neg part
                conv2_neg = tf.layers.conv2d(
                        inputs=level1_pooled_h_neg[i],  # (BS, q_len, d_len, 1) channel_last
                        filters=self.n_filters2[i],
                        kernel_size=self.kernel_sizes2[i],
                        padding='valid',
                        strides=self.conv_strides2[i],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                    uniform=True, seed=None, dtype=tf.float32),
                        kernel_regularizer=L2_regularizer,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        bias_regularizer=L2_regularizer,
                        name=None,
                        reuse=True)
                pool2_neg = tf.layers.max_pooling2d(
                        inputs=conv2_neg,
                        pool_size=self.pool_sizes2[i],
                        strides=self.pool_strides[i],
                        name=None)
                level2_pooled_h_neg.append(pool2_neg)
                M3_pos_L.append(MaxM_from4D(conv2_pos))  #ngramsets *(BS,)
                M3_neg_L.append(MaxM_from4D(conv2_neg))  #ngramsets *(BS,)
        M3_pos = tf.stack(M3_pos_L, axis=1)  # (BS, ngramsets)
        M3_neg = tf.stack(M3_neg_L, axis=1)  # (BS, ngramsets)
        M3_pos = tf.reduce_mean(M3_pos, axis=1)  # (BS,)
        M3_neg = tf.reduce_mean(M3_neg, axis=1)  # (BS,)
        ### MLP for level1
        ## max_pooling for the input interaction matrix
        with tf.variable_scope('maxpooling_interact_mat'):
            interact_mat1_pos
            pool0_pos = tf.layers.max_pooling2d(
                    inputs=input_mat1_pos,  # (BS, q_len, d_len, 1)
                    pool_size=self.pool_sizes0[0],
                    strides=self.pool_strides[0],
                    name=None)
            pool0_neg = tf.layers.max_pooling2d(
                    inputs=input_mat1_neg,  # (BS, q_len, d_len, 1)
                    pool_size=self.pool_sizes0[0],
                    strides=self.pool_strides[0],
                    name=None)
        # flatten
        last_pooled_flat_pos0 = tf.contrib.layers.flatten(pool0_pos)
        last_pooled_flat_neg0 = tf.contrib.layers.flatten(pool0_neg)
        mlp_h_pos1 = [last_pooled_flat_pos0]  # to store mlp hidden layers
        mlp_h_neg1 = [last_pooled_flat_neg0]
        for j in range(self.n_mlp_layers):
            scope_name = "mlp1_{}".format(j)
            with tf.variable_scope(scope_name):
                # pos part
                h_pos1 = tf.layers.dense(
                    inputs=mlp_h_pos1[j],
                    units=self.hidden_sizes[j],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=L2_regularizer,
                    bias_regularizer=L2_regularizer,
                    name=None,
                    reuse=False)
                mlp_h_pos1.append(h_pos1)
                # neg part
                h_neg1 = tf.layers.dense(
                    inputs=mlp_h_neg1[j],
                    units=self.hidden_sizes[j],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=L2_regularizer,
                    bias_regularizer=L2_regularizer,
                    name=None,
                    reuse=True)
                mlp_h_neg1.append(h_neg1)
        ### MLP for level 2
        last_pooled_flat_pos1_list = []
        last_pooled_flat_neg1_list = []
        for i in range(self.n_gramsets):
            last_pooled_flat_pos1_list.append(tf.contrib.layers.flatten(level1_pooled_h_pos[i]))
            last_pooled_flat_neg1_list.append(tf.contrib.layers.flatten(level1_pooled_h_neg[i]))
        last_pooled_flat_pos1 = tf.concat(last_pooled_flat_pos1_list, axis=1)
        last_pooled_flat_neg1 = tf.concat(last_pooled_flat_neg1_list, axis=1)
        mlp_h_pos2 = [last_pooled_flat_pos1]  # to store mlp hidden layers
        mlp_h_neg2 = [last_pooled_flat_neg1]
        for j in range(self.n_mlp_layers):
            scope_name = "mlp2_{}".format(j)
            with tf.variable_scope(scope_name):
                # pos part
                h_pos2 = tf.layers.dense(
                    inputs=mlp_h_pos2[j],
                    units=self.hidden_sizes[j],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=L2_regularizer,
                    bias_regularizer=L2_regularizer,
                    name=None,
                    reuse=False)
                mlp_h_pos2.append(h_pos2)
                # neg part
                h_neg2 = tf.layers.dense(
                    inputs=mlp_h_neg2[j],
                    units=self.hidden_sizes[j],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=L2_regularizer,
                    bias_regularizer=L2_regularizer,
                    name=None,
                    reuse=True)
                mlp_h_neg2.append(h_neg2)
        ### MLP for level 3
        last_pooled_flat_pos2_list = []
        last_pooled_flat_neg2_list = []
        for i in range(self.n_gramsets):
            last_pooled_flat_pos2_list.append(tf.contrib.layers.flatten(level2_pooled_h_pos[i]))
            last_pooled_flat_neg2_list.append(tf.contrib.layers.flatten(level2_pooled_h_neg[i]))
        last_pooled_flat_pos2 = tf.concat(last_pooled_flat_pos2_list, axis=1)
        last_pooled_flat_neg2 = tf.concat(last_pooled_flat_neg2_list, axis=1)
        mlp_h_pos3 = [last_pooled_flat_pos2]  # to store mlp hidden layers
        mlp_h_neg3 = [last_pooled_flat_neg2]
        for j in range(self.n_mlp_layers):
            scope_name = "mlp3_{}".format(j)
            with tf.variable_scope(scope_name):
                # pos part
                h_pos3 = tf.layers.dense(
                    inputs=mlp_h_pos3[j],
                    units=self.hidden_sizes[j],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=L2_regularizer,
                    bias_regularizer=L2_regularizer,
                    name=None,
                    reuse=False)
                mlp_h_pos3.append(h_pos3)
                # neg part
                h_neg3 = tf.layers.dense(
                    inputs=mlp_h_neg3[j],
                    units=self.hidden_sizes[j],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=L2_regularizer,
                    bias_regularizer=L2_regularizer,
                    name=None,
                    reuse=True)
                mlp_h_neg3.append(h_neg3)

        ### last scoring layer for level1
        with tf.variable_scope("output_layers1"):
            R_pos1 = tf.layers.dense(  # out (BS, 1)
                  inputs=mlp_h_pos1[-1],
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=False)
            R_neg1 = tf.layers.dense(  # out (BS, 1)
                  inputs=mlp_h_neg1[-1],
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=True)
        ### last scoring layer for level2
        with tf.variable_scope("output_layers2"):
            R_pos2 = tf.layers.dense(  # out (BS, 1)
                  inputs=mlp_h_pos2[-1],
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=False)
            R_neg2 = tf.layers.dense(  # out (BS, 1)
                  inputs=mlp_h_neg2[-1],
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=True)
        ### last scoring layer for level3
        with tf.variable_scope("output_layers3"):
            R_pos3 = tf.layers.dense(  # out (BS, 1)
                  inputs=mlp_h_pos3[-1],
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=False)
            R_neg3 = tf.layers.dense(  # out (BS, 1)
                  inputs=mlp_h_neg3[-1],
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=True)
        ## gating to aggregate
        M_pos = tf.stack([M1_pos, M2_pos, M3_pos], axis=1)  # (BS, 3)
        M_neg = tf.stack([M1_neg, M2_neg, M3_neg], axis=1)  # (BS, 3)
        beta_pos = tf.nn.softmax(self.gate * M_pos)  # (BS,3)
        beta_neg = tf.nn.softmax(self.gate * M_neg)  # (BS,3)

        ######## last incorporation layer for all granularities
        # concatenate for 3 Rs
        R_pos_input = tf.concat([R_pos1, R_pos2, R_pos3], axis=1)  # (BS, 3)
        R_neg_input = tf.concat([R_neg1, R_neg2, R_neg3], axis=1)  # (BS, 3)
        R_pos_input = beta_pos * R_pos_input
        R_neg_input = beta_neg * R_neg_input

        with tf.variable_scope("final_output"):
            R_pos = tf.layers.dense(  # out (BS, 1)
                  inputs=R_pos_input,
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=False)
            R_neg = tf.layers.dense(  # out (BS, 1)
                  inputs=R_neg_input,
                  units=1,
                  activation=None,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, seed=None, dtype=tf.float32),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=L2_regularizer,
                  bias_regularizer=L2_regularizer,
                  name=None,
                  reuse=True)

        return R_pos[:, 0], R_neg[:, 0]
