# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:07:39 2017

@author: maxime
"""

import numpy as np
import tensorflow as tf

def conv2d(input_, output_dim, k_h, k_w, name="conv2d", reuse=None):
    # cnn_inputs, kernel_feature_size, 1, kernel_size
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def char_convolution(inputs, kernels, kernel_features, scope="char_convolution", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Given an input of shape [batch, s_length, w_length, embed_dim]
        # performs a 1D convolutions with multiple kernel sizes.
        s_length = tf.shape(inputs)[1]
        w_length = tf.shape(inputs)[2]
        e_length = tf.shape(inputs)[3]
        cnn_inputs = tf.reshape(inputs, [batch_size*s_length, w_length, e_length])
        # Expand the second dimension for convolution purposes
        cnn_inputs = tf.expand_dims(cnn_inputs, 1)
        # Layer to hold all of the convolution results
        layers = []
        # For each kernel, tuple of [kernel size, num filters]
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            # Apply the convolution on all of the inputs for this kernal
            conv = conv2d(cnn_inputs, kernel_feature_size, 1, kernel_size,
                          name="kernel_%d" % kernel_size, reuse=reuse)
            # Take the max pooling
            pool = tf.reduce_max(tf.tanh(conv), 2, keep_dims=True)
            # Append the squeezed version
            layers.append(tf.squeeze(pool, [1, 2]))
        # Concat the results along the second axis to give words embeddings
        cnn_output = tf.concat(layers, 1)
        # Reshape it to match sequence length
        cnn_output = tf.reshape(cnn_output, [batch_size, timesteps, sum(kernel_features)])
        return cnn_output

def positional_encoding(inputs, max_len, num_units, scope="pos_embedding", reuse=None):
    """Returns the sinusoidal encodings matrix
    inputs: matrix with the inputs
    max_len: the max timesteps of the model
    num_units: the dimension to encode every timestep in (512)
    """
    with tf.variable_scope(scope, reuse=reuse)
        # Variable like [1, 2, ... max_len]*batch_size
        input_one = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0),
                                                    [tf.shape(inputs)[0], 1])
        position_enc = np.array([
            [pos / np.power(10000, 2*i/num_units) for i in range(num_units)]
            for pos in range(max_len)])

        position_enc[:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc)
        outputs = tf.nn.embedding_lookup(lookup_table, input_one)
        outputs = outputs * math.sqrt(num_units)
        return outputs

def embedding(inputs, vocab_size, num_units, scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=True)
        # Variable that holds the mapping between the ids and their representation
        lookup_table = tf.get_variable('lookup_table',
                           dtype=tf.float32,
                           shape=[vocab_size, num_units],
                           initializer=tf.contrib.layers.xavier_initializer())
        # Change the first row with a tensor full of zeros (PAD id)
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                          lookup_table[1:, :]), 0)
        # Return the embeddings
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        # Scale it with the square root of num_units
        return outputs * (num_units ** 0.5)

def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout=0,
                        causality=False, scope="multihead_attention",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse)
        num_units = queries.get_shape()[-1]
        projection_dim = num_units / num_heads
        # Variable to store the intermediate multi-head attention
        intermediate_outputs = []
        # For every head
        for i in range(num_heads):
            # Project the queries, keys and values on projection_dim
            Q = tf.layers.dense(queries, projection_dim)
            K = tf.layers.dense(keys, projection_dim)
            V = tf.layers.dense(keys, projection_dim)

            # Multiply the query with the keys to get the attention scores
            scores = tf.matmul(Q, tf.transpose(K))

            # Scale the result
            scaled_scores = scores / (K.get_shape()[-1] ** 0.5)

            # Mask the scores that are 'illegal'
            # Causality = Future blinding
            if causality:
                # Create a vector of ones that has the same size as the self
                # projection for one batch
                diag_vals = tf.ones_like(scaled_scores[0, :, :]) # (T_q, T_k)
                # Create a lower triangular matrix with it.
                # The first row will be equal to one only for the first item
                # So all the other attention will be set to minus infinity after
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
                # Expand the mask for the batch_size
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(scaled_scores)[0], 1, 1])
                # Padding is the value of initinity that will be used when the
                # attentionw will be illegal
                paddings = tf.ones_like(masks)*(-2**32+1)
                # Wheneve the mask equal one, replace the score value with the value
                # of the padding (-inf)
                scaled_scores = tf.where(tf.equal(masks, 0), paddings, scaled_scores) # (h*N, T_q, T_k)

            # Perform the softmax
            softmax = tf.nn.softmax(scaled_scores)
            # Apply dropout
            dropout = tf.layers.dropout(softmax, rate=dropout)
            # Compute the attention (finally)
            outputs = tf.matmul(dropout, V)
            # Append the result
            intermediate_outputs.append(outputs)
        # Out of the loop
        # Concat the intermediate results
        concat = tf.concat(intermediate_outputs, axis=2)
        # Apply a last linear projection
        linear = tf.layers.dense(concat, num_units, activation=tf.nn.relu)
        # Residual connection (Add)
        outputs = queries + linear
        # Normalize outputs (Norm)
        outputs = tf.contrib.layers.layer_norm(outputs)
        return linear

def feed_forward(inputs, scope="multihead_attention", reuse=None):
    # This layer needs to be seen as simple dense layer that is applied
    # element per element (i.e., word per word), just like ... a 1D convolution
    # on the inputs
    # This layer has two part. First one a element dense layer of size 22048
    # with a relu activation function
    with tf.variable_scope(scope, reuse=reuse):
        params = {"inputs": inputs, "filters": 2048, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # And a second layer element wise dense layer that project the output
        # to the num_units of 512
        params = {"inputs": outputs, "filters": 512, "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection (Add)
        outputs += inputs
        # Normalize outputs (Norm)
        outputs = tf.contrib.layers.layer_norm(outputs)
        return outputs

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape()[-1]
    return ((1-epsilon) * inputs) + (epsilon / K)

def normalize(inputs, epsilon=1e-8):
