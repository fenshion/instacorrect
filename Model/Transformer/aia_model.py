# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:39:53 2017

@author: maxime
"""

import tensorflow as tf
from aia_modules import char_convolution, positional_encoding, multihead_attention, feedforward

def aiamodel(features, labels, mode, params):
    
    with tf.variable_scope('ModelParams'):
        batch_size = tf.shape(features['sequence'])[0]
        timesteps = tf.shape(features['sequence'])[1]
        maxwordlength = tf.shape(features['sequence'])[2]
        c_embed_s = params['char_embedding_size']
        dropout = params['d ropout']
        hidden_size = params['hidden_size']
        network_depth = params['network_depth']
        kernels = params['kernels']
        kernel_features = params['kernel_features']
        ultimate_sequ_len = params['ultimate_sequ_len']
        
    with tf.variable_scope('Convolution'):
        # Characters embeddings matrix. Basically each character id (int)
        # is associated a vector [char_embedding_size]
        embeddings_c = tf.Variable(tf.random_uniform([params['char_vocab_size'], 
                                   c_embed_s], -1.0, 1.0))
        # Embed every char id into their embedding. Will go from this dimension
        # [batch_size, max_sequence_length, max_word_size] to this dimension
        # [batch_size, max_sequence_length, max_word_size, char_embedding_size]
        embedded_inputs = tf.nn.embedding_lookup(embeddings_c, features['sequence'])
        # Apply a character convolution on the characters
        conv_inputs = char_convolution(embedded_inputs, kernels, kernel_features,
                                        reuse=True)
        # Map the result to the 512 dimension
        conv_inputs = tf.layers.dense(conv_inputs, 512)
        
        embedded_outputs = tf.nn.embedding_lookup(embeddings_c, labels['sequence'])
        # Apply a character convolution on the characters
        conv_outputs = char_convolution(embedded_outputs, kernels, kernel_features,
                                        reuse=True)
        # Map the result to the 512 dimension
        conv_outputs = tf.layers.dense(conv_outputs, 512)
    
    with tf.variable_scope('Encoder'):
        position_embeddings = positional_encoding(conv_outputs, 
                                                  ultimate_sequ_len, 512)
        encoder_inputs = position_embeddings + conv_outputs
        encoder_inputs = tf.layers.dropout(encoder_inputs, rate=dropout)
        
        for i in range(params['num_blocks']):
            encoder_inputs = multihead_attention(queries=encoder_inputs,
                                                 keys=encoder_inputs,
                                                 num_units=512,
                                                 num_heads=8,
                                                 dropout=dropout,
                                                 causality=False)
            encoder_inputs = feedforward(encoder_inputs)
        
    with tf.variable_scope('Decoder'):
        # Decoder
        if mode != tf.estimator.ModeKeys.PREDICT: 
            # Training time so the decoder inputs are available, they wont be
            # available at inference time.
            # Should replace the first character of every sequence with the
            # go character. output is of the shape [batc, seqlen, wordlen]
            decoder_outputs = labels['sequence']
            shape_deco = tf.shape(decoder_outputs)
            go_char = tf.fill([shape_deco[0], 1, shape_deco[2]], params['go_char'])
            decoder_inputs = tf.concat([go_char, decoder_outputs[:, :-1, :]], axis=2)
            
            embedded_outputs = tf.nn.embedding_lookup(embeddings_c, decoder_inputs)
            # Apply a character convolution on the characters
            decoder_inputs_e = char_convolution(embedded_outputs, kernels, kernel_features,
                                            reuse=True)
            # Map the result to the 512 dimension
            decoder_inputs_e = tf.layers.dense(decoder_inputs, 512)
            output_pos_embbed = positional_encoding(decoder_inputs_e, 
                                                  ultimate_sequ_len, 512)
            decoder_inputs += output_pos_embbed
            decoder_inputs = tf.layers.dropout(decoder_inputs, rate=dropout)