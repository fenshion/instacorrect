# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:39:53 2017

@author: maxime
"""

import tensorflow as tf
from aia_modules import char_convolution, positional_encoding, multihead_attention, feed_forward, decode_step, prepare_encoder_inputs, is_eos, positional_encoding_table

def aiamodel(features, labels, mode, params):
    """
    Tansformer model from "Attention is all you need"
    """
    with tf.variable_scope('ModelParams'):
        batch_size = tf.shape(features['sequence'])[0] # the current batch size
        timesteps = tf.shape(features['sequence'])[1] # The number of unrollings for the encoder
        maxwordlength = tf.shape(features['sequence'])[2] # Maximum length of words in the batch
        c_embed_s = params['char_embedding_size'] # The embedding size of characters
        dropout = params['d ropout']
        hidden_size = params['hidden_size']
        network_depth = params['network_depth']
        kernels = params['kernels']
        kernel_features = params['kernel_features']
        ultimate_sequ_len = params['ultimate_sequ_len']
        position_embeddings_table = positional_encoding_table(ultimate_sequ_len, 512)

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

    with tf.variable_scope('Encoder'):
        # Positional Embeddings
        input_one = tf.tile(tf.expand_dims(tf.range(timesteps), 0), [batch_size, 1])
        position_emb = tf.nn.embedding_lookup(position_embeddings_table, input_one)
        position_emb = position_emb * math.sqrt(num_units)
        conv_outputs = position_emb + conv_outputs
        
        # Encode inputs
        encoder_inputs = tf.layers.dropout(conv_outputs, rate=dropout)
        for i in range(params['num_blocks']):
            encoder_inputs = multihead_attention(queries=encoder_inputs,
                                                 keys=encoder_inputs,
                                                 num_units=512,
                                                 num_heads=8,
                                                 dropout=dropout,
                                                 causality=False)
            encoder_inputs = feed_forward(encoder_inputs)
        
        # Encoder outputs    
        encoder_outputs = encoder_inputs

    with tf.variable_scope('Decoder'):
        # Decoder
        # Training time so the decoder inputs are available, they wont be
        # available at inference time.
        # Should replace the first character of every sequence with the
        # go character. output is of the shape [batc, seqlen, wordlen]
        if mode == tf.estimator.ModeKeys.PREDICT:
            go_char = tf.fill([batch_size, 1, 10], params['go_char'])
            decoder_inputs = go_char
            decoder_inputs = prepare_encoder_inputs(decoder_inputs, kernels, kernel_features,
                                                    embeddings_c, ultimate_sequ_len)
            def greedy_infer_step(encoder_outputs, i):
                """ Perfom a one step greedy inference step """
                decoder_output = decode_step(encoder_outputs, decoder_input, num_blocks)
                logits = tf.layers.dense(decoder_output, vocab_size)
                preds = tf.cast(tf.arg_max(logits, dimension=-1), tf.int32)
                # Check if EOS
                rez = tf.cond(is_eos(preds, eos, batch_size), tf.add(i, 10000), tf.add(i, 1))
                # Check in the preds if there all the sentences are EOS. If yes assign +inf to i
                # else increment i by one.
                # Take the preds and find their char rep in the hashtable
                # Convert the sparse tensor from the hashtable to a dense representation.
                # then prepare_encoder_inputs with the dense tensor
                # This is the new deocder_inputs
                values = table.lookup(preds)
                # We now have an array of ['1,44,54','4,66,5']
                splited = tf.string_split(values, ',')
                # Sparse tensor with the splited strings as values
                splited_int = tf.string_to_number(splited.values, out_type=tf.int32)
                # Make the values to int32 and convert to dense
                preds_chars = tf.sparse_to_dense(splited.indices, splited.dense_shape, splited_int)
                # Send the characters to embeddings.
                preds_embed = tf.nn.embedding_lookup(embeddings_c, preds_chars)
                # Augment the second dimension (sequence of length 1)
                preds_embed = tf.expand_dims(preds_embed, 1)
                # Send for convolution
                preds_convo = char_convolution(preds_embed, kernels, kernel_features, reuse=True)
                # Delete the second dimension
                preds_convo = tf.squeeze(preds_convon, axis=1)
                # Position Embeddings
                input_one = tf.tile(tf.expand_dims(i, 0), [batch_size, 1])
                position_emb = tf.nn.embedding_lookup(position_embeddings_table, input_one)
                position_emb = position_emb * math.sqrt(num_units)
                preds_output = preds_convo + position_emb
                # Concat this step's results with the previous steps results
                decoder_output = tf.concat([decoder_output, preds_output], axis=1)
                return decoder_output, i
            # TF WHILE
        else:
            # Training time
            decoder_outputs = labels['sequence']
            shape_deco = tf.shape(decoder_outputs)
            go_char = tf.fill([shape_deco[0], 1, shape_deco[2]], params['go_char'])
            decoder_inputs = tf.concat([go_char, decoder_outputs[:, :-1, :]], axis=2)
            decoder_inputs = prepare_encoder_inputs(decoder_inputs, kernels, kernel_features,
                                                    embeddings_c, ultimate_sequ_len)
            # Add dropout on the decoder inputs.
            decoder_inputs = tf.layers.dropout(decoder_inputs, rate=dropout)
            decoder_outputs = decode_step(encoder_outputs, decoder_inputs, params['num_blocks'])

    # Final layer projection
    with tf.variable_scope('final_layer'):
        logits = tf.layers.dense(decoder_outputs, params['word_vocab_size'])
        preds = tf.cast(tf.arg_max(self.logits, dimension=-1), tf.int32)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_o,
                                                              logits=logits)
        target_w = tf.sequence_mask(labels['sequence_length'], dtype=logits.dtype)
        batch_size_32 = tf.cast(batch_size, tf.float32)
        timesteps_32 = tf.cast(timesteps, tf.float32)
        loss = (tf.reduce_sum(crossent * target_w) / (batch_size_32+timesteps_32))
    # Train
    with tf.variable_scope('train'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                               tf.train.get_global_step(),
                                               params['decay_steps'],
                                               0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # Apply gradient clipping
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables),
                                                 global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Evaluation
    with tf.variable_scope('eval')
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=decoder_o,
                                                   predictions=sample_id,
                                                   weights=target_w)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
