# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:39:53 2017

@author: maxime
"""

import tensorflow as tf
from trans_modules import char_convolution, project_embedding, initializer
from trans_modules import multihead_attention, encode_positions, feed_forward
from trans_modules import decode_step, is_eos, positional_encoding_table, make_logits

import math


def transformer(features, labels, mode, params):
    """
    Tansformer model from "Attention is all you need"
    """
    with tf.variable_scope('VariableDefinition'):
        input_shape = tf.shape(features['sequence'])
        batch_size = input_shape[0]  # the current batch size
        timesteps = input_shape[1]  # The number of unrollings for the encoder
        c_embed_s = params['char_embedding_size']  # The embedding size of char
        w_embed_s = params['word_embedding_size']  # The embedding size of char
        vocab_size_char = params['char_vocab_size']
        vocab_size_word = params['word_vocab_size']
        dropout = params['dropout']
        hidden_size = params['hidden_size']
        kernels = params['kernels']
        kernel_features = params['kernel_features']
        ultimate_sequ_len = params['ultimate_sequ_len']
        num_blocks = params['num_blocks']
        num_heads = params['attention_heads']
        eos = params['eos_id']
        go = params['go_id']
        position_embed_table = positional_encoding_table(ultimate_sequ_len,
                                                         hidden_size)
        embeddings_c = tf.Variable(tf.random_uniform([vocab_size_char,
                                                      c_embed_s], -1.0, 1.0))
        embeddings_w = tf.Variable(tf.random_uniform([vocab_size_word,
                                                      w_embed_s], -1.0, 1.0))

    with tf.variable_scope('Convolution'):
        # Embed every char id into their embedding. Will go from this dimension
        # [batch_size, max_sequence_length, max_word_size, char_embedding_size]
        embedded_inputs = tf.nn.embedding_lookup(embeddings_c,
                                                 features['sequence'])
        # Apply a character convolution on the characters. Returns
        # [batch, seqlen, maxword, kernels*kernel_features]
        conv_inputs = char_convolution(embedded_inputs, kernels,
                                       kernel_features, c_embed_s, hidden_size,
                                       reuse=None)
        # Map the result to the 512 dimension
        conv_outputs = tf.layers.dense(conv_inputs, 512,
                                       kernel_initializer=initializer)
    with tf.variable_scope('Encoder'):
        # Positional Embeddings
        position_emb = encode_positions(position_embed_table, conv_outputs)
        conv_outputs = position_emb + conv_outputs
        # Encode inputs
        encoder_inputs = tf.layers.dropout(conv_outputs, rate=dropout)
        for i in range(num_blocks):  # Should equal 6 -> stacked identical layr
            scope = 'encoder_multihead_attention_{i}'.format(i=str(i))
            encoder_inputs = multihead_attention(queries=encoder_inputs,
                                                 keys=encoder_inputs,
                                                 num_units=hidden_size,
                                                 num_heads=num_heads,
                                                 dropout=dropout,
                                                 scope=scope,
                                                 causality=False)
            scope = 'encoder_feed_forward_{i}'.format(i=str(i))
            encoder_inputs = feed_forward(encoder_inputs, scope=scope)
        # Encoder outputs
        encoder_outputs = encoder_inputs
    with tf.variable_scope('Decoder'):
        # Decoder
        # Seperate for training and inference
        if mode == tf.estimator.ModeKeys.PREDICT:
            # We need to tile it to the correct_batch_size
            go_char = tf.tile(tf.reshape(go, [1, 1]), [batch_size, 1])
            # Embed the characters
            decoder_inputs = tf.nn.embedding_lookup(embeddings_w, go_char)
            decoder_inputs = project_embedding(decoder_inputs, hidden_size)
            # Get the positional embeddings
            output_pos_embbed = encode_positions(position_embed_table,
                                                 decoder_inputs)
            # # Add the positional embeddings
            decoder_inputs += output_pos_embbed

            def greedy_infer_step(i, decoder_inputs):
                """ Perfom a one step greedy inference step """
                previous_inputs = decoder_inputs
                decoder_inputs = decoder_inputs[:, -1, :]
                decoder_inputs = tf.expand_dims(decoder_inputs, axis=1)
                decoder_output = decode_step(encoder_outputs, decoder_inputs,
                                             num_blocks)
                logits = make_logits(decoder_output, vocab_size_word)
                preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
                # Check if EOS
                i = tf.cond(is_eos(preds, eos, batch_size),
                            lambda: tf.add(i, 10000),
                            lambda: tf.add(i, 1))
                i = tf.Print(i, [i], message="i")
                preds = tf.Print(preds, [preds], message="preds")
                # Check in the preds if there all the sentences are EOS.
                # If yes assign +inf to i else increment i by one.
                # Take the preds and find their char rep in the hashtable
                # Convert the sparse tensor from the hashtable to a dense repr.
                # then prepare_encoder_inputs with the dense tensor
                # This is the new deocder_inputs
                preds_embed = tf.nn.embedding_lookup(embeddings_w, preds)
                preds_embed = project_embedding(preds_embed, hidden_size,
                                                reuse=True)
                position_emb = encode_positions(position_embed_table,
                                                preds_embed,
                                                inference=True,
                                                i=i)
                preds_output = preds_embed + position_emb
                # Concat this step's results with the previous steps results
                decoder_output = tf.concat([previous_inputs, preds_output],
                                           axis=1)
                decoder_output = tf.Print(decoder_output, [tf.shape(decoder_output)], message="decoder_output")
                return i, decoder_output

            def condition(i, decoder_inputs):
                return tf.less(i, timesteps*2)
            i = tf.constant(0, dtype=tf.int32)
            tshape = [i.get_shape(), tf.TensorShape([None, None, hidden_size])]
            i, decoder_outputs = tf.while_loop(condition, greedy_infer_step,
                                               [i, decoder_inputs],
                                               shape_invariants=tshape)
        else:
            # Training time
            # Embed the word ids
            decoder_inpt = labels['sequence']
            # Create the decoder output by removing the first word id (go)
            decoder_o = decoder_inpt[:, 1:]
            decoder_i = decoder_inpt[:, :-1]
            decoder_i = tf.nn.embedding_lookup(embeddings_w, decoder_i)
            decoder_i = project_embedding(decoder_i, hidden_size)
            # Get the positional embeddings
            output_position_emb = encode_positions(position_embed_table,
                                                   decoder_i)
            # Add the positional embeddings
            decoder_i += output_position_emb
            # Add dropout on the decoder inputs.
            decoder_i = tf.layers.dropout(decoder_i, rate=dropout)
            decoder_outputs = decode_step(encoder_outputs, decoder_i,
                                          params['num_blocks'])
        # End of IF
        # Project the decoder output to a word_vocab_size dimension
        if mode == tf.estimator.ModeKeys.PREDICT:
            reuse = True
        else:
            reuse = False
        logits = make_logits(decoder_outputs, vocab_size_word, reuse=reuse)
        # Take the arg max for each, ie, the word id.
        preds = tf.cast(tf.argmax(logits, axis=2), tf.int32)
        # preds = tf.Print(preds, [preds[0, :]], message="preds1", summarize=10)
        # preds = tf.Print(preds, [logits[0, :]], message="logits", summarize=10)
        # preds = tf.Print(preds, [decoder_inpt[0, :]], message="decoder_i1", summarize=10, first_n=1)
        # preds = tf.Print(preds, [decoder_o[0, :]], message="decoder_o1", summarize=10, first_n=1)
        # preds = tf.Print(preds, [preds[1, :]], message="preds2", summarize=10)
        # preds = tf.Print(preds, [decoder_inpt[1, :]], message="decoder_i2", summarize=10, first_n=1)
        # preds = tf.Print(preds, [decoder_o[1, :]], message="decoder_o2", summarize=10, first_n=1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Return a dict with the sample word ids.
            preds = {'sequence': preds}
            export = {
                'prediction': tf.estimator.export.PredictOutput(preds)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=preds,
                                              export_outputs=export)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_o,
                                                              logits=logits)
        # Create a mask for padding characters
        target_w = tf.sequence_mask(labels['sequence_length'],
                                    dtype=tf.float32)
        batch_size_32 = tf.cast(batch_size, tf.float32)
        timesteps_32 = tf.cast(tf.reduce_sum(timesteps), tf.float32)
        # Apply the mask and normalize the loss for the batch size. Could also
        # Normalize the loss for the sequence length
        loss = (tf.reduce_sum(loss * target_w) / (batch_size_32*timesteps_32))
    # Train
    with tf.variable_scope('train'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = tf.train.exponential_decay(params['learning_rate'],
                                            tf.train.get_global_step(),
                                            params['decay_steps'],
                                            0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)
            # Apply gradient clipping
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            gstep = tf.train.get_global_step()
            train_op = optimizer.apply_gradients(zip(gradients, variables),
                                                 global_step=gstep)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              train_op=train_op)
    # Evaluation
    with tf.variable_scope('eval'):
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=decoder_o,
                                                           predictions=preds,
                                                           weights=target_w)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)
