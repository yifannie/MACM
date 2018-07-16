import numpy as np
import tensorflow as tf

def masked_softmax(x, axis=-1, mask=None):
    mask = tf.cast(mask, x.dtype)
    if mask is not None:
        x = (mask * x) + (1 - mask) * (-10)
    x = tf.clip_by_value(x, -10, 10)
    e_x = tf.exp(x - tf.reduce_max(x, axis=axis, keep_dims=True))
    if mask is not None:
        e_x = e_x * mask
    softmax = e_x / (tf.reduce_sum(e_x, axis=axis, keep_dims=True) + 1e-6)
    return softmax

def np_softmax(x, axis=-1):
    # stable softmax for np array
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    softmax = e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-9)
    return softmax

def non_neg_normalize(x):
    """ input: a np array vector
    output: (x - x_min)/(x_max - x_min)
    """
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-6)

def non_neg_normalize_2Darray(x):
    """input: np 2D array
    output column-wise (x - x_min)/(x_max - x_min)
    """
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-6)

def cossim(x, y):
    '''input 2 tf placeholder vectors
       returns the cosine similarity of the 2 vectors
    '''
    norm_x = tf.sqrt(tf.reduce_sum(x ** 2, axis=0))
    norm_y = tf.reduce_sum(tf.reduce_sum(y ** 2, axis=0))
    return tf.reduce_sum(x * y, axis=0) / (norm_x * norm_y + 1e-5)

def batch_cossim(X, Y):
    '''batch version of cossim, by batch matmul
       X placeholder tensor of shape (batchsize, len1, emb_dim)
       Y placeholder tensor of shape (batchsize, len2, emb_dim)
       returns: the cos similarity interaction tensor
       of shape (batchsize, len1, len2)
    '''
    norm_X = tf.expand_dims(tf.sqrt(tf.reduce_sum(X ** 2, axis=2)), axis=2)  # (BS, len1, 1)
    norm_Y = tf.expand_dims(tf.sqrt(tf.reduce_sum(Y ** 2, axis=2)), axis=1)  # (BS, 1, len2)
    scalar_prod = tf.matmul(X, tf.transpose(Y, perm=[0,2,1]))  # (BS, len1, len2)
    return scalar_prod * 1.0 / (norm_X * norm_Y + 1e-5)   # stable

def batch_grid_topk(X, k, axis=2):
    '''batch version of graid_topk
    X: placeholder of a similarity grid (batchsize, q_len, d_len)
    k: keep top k cols or rows of X and preserve ordering
    axis = 1 or 2, 1=drop less intense q terms, 2=drop less intensive d terms
    returns: donwsized interaction grid
    '''
    if axis == 2:  # drop doc terms
        max_values = tf.reduce_max(X, axis=1)  # (BS, d_len) doc term max interact values
        topk = tf.nn.top_k(max_values, k)  # topk.values=(BS, k) topk.index=(BS,k)
        kth_elements = tf.reduce_min(topk.values, axis=1) # (BS,)
        mask = tf.greater_equal(max_values, tf.expand_dims(kth, axis=1))  # (BS, d_len), line format [true, false, true ..]


    if axis == 1:  # drop query terms
        topk = tf.nn.top_k(max_values, k)  # topk.values=(BS, k) topk.index=(BS,k)
        kth_elements = tf.reduce_min(topk.values, axis=1) # (BS,)
        mask = tf.greater_equal(max_values, tf.expand_dims(kth, axis=1))  # (BS, q_len), line format [true, false, true ..]

    else:
        raise ValueError("axis must be 1(q_len) or 2(doc_len), not the batchsize axe=0")

def batch_MaxM(X):
    """ calculate max M for a batch of interact grid (BS, q_len, d_len)
    returns a scalar M
    """
    # for each query term (row), find the max interact intensity across all doc terms
    M = tf.reduce_max(X, axis=2)  # (BS, q_len)
    # calcuate the sum of the Ms to represent the M of the whole query
    M = tf.reduce_sum(M, axis=1)  # (BS,)
    # take avg across all docs in this batch to get generalization capability
    M = tf.reduce_mean(M)  # (1,)
    return M

def BatchFeatMap_MaxM(X):
    """ calculate max M for a batch and feature maps
     of interact grid (BS, q_len, d_len, nfeatmaps)
    returns a scalar M
    """
    M = tf.reduce_max(X, axis=2)  # (BS, qlen, nfeatmaps)
    M = tf.reduce_sum(M, axis=1)  #(BS, nfeatmaps)
    M = tf.reduce_mean(M, axis=0)  #(nfeatmaps, )
    M = tf.reduce_mean(M)
    return M

def MaxM_fromBatch(X):
    """ calculate max M for a batch of interact grid (BS, q_len, d_len)
    returns a vector M (BS, ) for the batch including each instance's M
    """
    # for each query term (row), find the max interact intensity across all doc terms
    M = tf.reduce_max(X, axis=2)  # (BS, q_len)
    # calcuate the sum of the Ms to represent the M of the whole query
    M = tf.reduce_sum(M, axis=1)  # (BS,)
    return M

def MaxM_from4D(X):
    """ calculate max M for a batch and feature maps
     of interact grid (BS, q_len, d_len, nfeatmaps)
    returns a vector M (BS, ) for the batch including each instance's
    """
    M = tf.reduce_max(X, axis=2)  # (BS, qlen, nfeatmaps)
    M = tf.reduce_sum(M, axis=1)  #(BS, nfeatmaps)
    M = tf.reduce_mean(M, axis=1)  #(BS, )
    return M

def threshold_1dir(X, value):
    """ calculate a element-wise thresholded version of the input tensor X
    X: input tensor
    value: float value of the threshold
    return X * (X >= value)
    """
    return X * tf.cast(X >= value, tf.float32)

def threshold(X, value):
    """ calculate a element-wise 2-way thresholded version of the input tensor X
    X: input tensor
    value: float value of the threshold
    return X * (X >= value or X <= -value)
    """
    X1 = tf.cast(X >= value, tf.float32)
    X2 = tf.cast(X <= -value, tf.float32)
    return X * tf.cast( (X1 + X2) > 0, tf.float32)

def rescale(X, target_min, target_max, mode='linear'):
    """ rescales the elements in tensor X to range (X_min, X_max)
        X: input tensor (BS, xlen, ylen)
        X_min: minval after rescale
        X_max: maxval after rescale
        mode: scaling function
        returns: rescaled X
    """
    X_min = tf.reduce_min(X, axis=[1, 2])  # (BS, )
    X_max = tf.reduce_max(X, axis=[1, 2])  # (BS, )
    if mode == "linear":
        X = (X - tf.expand_dims(tf.expand_dims(X_min, axis=1), axis=1)) / \
            (tf.expand_dims(tf.expand_dims(X_max, axis=1), axis=1) - tf.expand_dims(tf.expand_dims(X_min, axis=1), axis=1)
             + 1e-6)
        # rescale to target range
        X = X * (target_max - target_min) + target_min
    if mode == "tanh":
        pass
    return X

def rescale_and_threshold(X, target_min, target_max, thres, mode='linear'):
    """ first rescale and then filter out weak interactions
    """
    if mode == "linear":
        X = rescale(X, target_min, target_max, mode=mode)
        X = threshold(X, thres)
    return X
