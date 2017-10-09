# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:39:53 2017

@author: maxime
"""

import tensorflow as tf
from trans_modules import char_convolution, make_table
from trans_modules import multihead_attention, encode_positions, feed_forward
from trans_modules import decode_step, is_eos, positional_encoding_table
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
        vocab_size_char = params['char_vocab_size']
        vocab_size_word = params['word_vocab_size']
        dropout = params['dropout']
        hidden_size = params['hidden_size']
        kernels = params['kernels']
        kernel_features = params['kernel_features']
        ultimate_sequ_len = params['ultimate_sequ_len']
        num_blocks = params['num_blocks']
        num_heads = params['attention_heads']
        eos = params['eos']
        position_embed_table = positional_encoding_table(ultimate_sequ_len,
                                                         hidden_size)
        embeddings_c = tf.Variable(tf.random_uniform([vocab_size_char,
                                                      c_embed_s], -1.0, 1.0))

    with tf.variable_scope('Convolution') as convolution:
        # Embed every char id into their embedding. Will go from this dimension
        # [batch_size, max_sequence_length, max_word_size] to this dimension
        # [batch_size, max_sequence_length, max_word_size, char_embedding_size]
        embedded_inputs = tf.nn.embedding_lookup(embeddings_c,
                                                 features['sequence'])
        # Apply a character convolution on the characters
        conv_inputs = char_convolution(embedded_inputs, kernels,
                                       kernel_features, c_embed_s, reuse=None)
        # Map the result to the 512 dimension
        conv_outputs = tf.layers.dense(conv_inputs, 512)
        print('conv_outputs', conv_outputs)
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
        # Training time so the decoder inputs are available, they wont be
        # available at inference time.
        # Should replace the first character of every sequence with the
        # go character. output is of the shape [batc, seqlen, wordlen]
        if mode == tf.estimator.ModeKeys.PREDICT:
            go_char = tf.fill([batch_size, 1, 10], params['go_char'])
            # Embed the characters
            decoder_inputs = tf.nn.embedding_lookup(embeddings_c, go_char)
            # Apply a character convolution on the characters
            with tf.variable_scope(convolution, reuse=True):
                decoder_inputs = char_convolution(decoder_inputs, kernels,
                                                  kernel_features, c_embed_s,
                                                  reuse=True)
            # Map the result to the 512 dimension
            decoder_inputs = tf.layers.dense(decoder_inputs, 512)
            # Get the positional embeddings
            output_pos_embbed = encode_positions(position_embed_table,
                                                 decoder_inputs)
            # # Add the positional embeddings
            decoder_inputs += output_pos_embbed
            # Table to get every words characters at decode time
            table = make_table(params['words_vocab_filename'])

            def greedy_infer_step(decoder_inputs, i):
                """ Perfom a one step greedy inference step """
                decoder_output = decode_step(encoder_outputs, decoder_inputs,
                                             num_blocks, reuse=True)
                logits = tf.layers.dense(decoder_output, vocab_size_word)
                preds = tf.cast(tf.arg_max(logits, dimension=-1), tf.int32)
                # Check if EOS
                tf.cond(is_eos(preds, eos, batch_size), tf.add(i, 10000),
                        tf.add(i, 1))
                # Check in the preds if there all the sentences are EOS.
                # If yes assign +inf to i else increment i by one.
                # Take the preds and find their char rep in the hashtable
                # Convert the sparse tensor from the hashtable to a dense repr.
                # then prepare_encoder_inputs with the dense tensor
                # This is the new deocder_inputs
                values = table.lookup(preds)
                # We now have an array of ['1,44,54','4,66,5']
                splited = tf.string_split(values, ',')
                # Sparse tensor with the splited strings as values
                splited_int = tf.string_to_number(splited.values,
                                                  out_type=tf.int32)
                # Make the values to int32 and convert to dense
                preds_chars = tf.sparse_to_dense(splited.indices,
                                                 splited.dense_shape,
                                                 splited_int)
                # Send the characters to embeddings.
                preds_embed = tf.nn.embedding_lookup(embeddings_c, preds_chars)
                # Augment the second dimension (sequence of length 1)
                preds_embed = tf.expand_dims(preds_embed, 1)
                # Send for convolution
                with tf.variable_scope(convolution, reuse=True):
                    preds_convo = char_convolution(preds_embed, kernels,
                                                   kernel_features, c_embed_s,
                                                   reuse=True)
                # Delete the second dimension
                preds_convo = tf.squeeze(preds_convo, axis=1)
                # Position Embeddings
                input_one = tf.tile(tf.expand_dims(i, 0), [batch_size, 1])
                position_emb = tf.nn.embedding_lookup(position_embed_table,
                                                      input_one)
                position_emb = position_emb * math.sqrt(hidden_size)
                preds_output = preds_convo + position_emb
                # Concat this step's results with the previous steps results
                decoder_output = tf.concat([decoder_output, preds_output],
                                           axis=1)
                return decoder_output, i

            def condition(decoder_inputs, i):
                tf.less(i, timesteps*2)
            i = tf.constant(0)
            tshape = [tf.TensorShape([batch_size, None, None]), i.get_shape()]
            decoder_outputs = tf.while_loop(condition, greedy_infer_step,
                                            [decoder_inputs, i],
                                            shape_invesriants=tshape)
        else:
            # Training time
            # Embed the characters
            decoder_inpt = labels['sequence_chars']
            decoder_inputs = tf.nn.embedding_lookup(embeddings_c, decoder_inpt)
            # Apply a character convolution on the characters
            with tf.variable_scope(convolution, reuse=True):
                decoder_inputs = char_convolution(decoder_inputs, kernels,
                                                  kernel_features, c_embed_s,
                                                  reuse=True)
            # Map the result to the 512 dimension
            decoder_inputs = tf.layers.dense(decoder_inputs, hidden_size)
            # Get the positional embeddings
            output_position_emb = encode_positions(position_embed_table,
                                                   decoder_inputs)
            # # Add the positional embeddings
            decoder_inputs += output_position_emb
            # Add dropout on the decoder inputs.
            decoder_inputs = tf.layers.dropout(decoder_inputs, rate=dropout)
            decoder_outputs = decode_step(encoder_outputs, decoder_inputs,
                                          params['num_blocks'])

    # Final layer projection
    with tf.variable_scope('final_layer'):
        # Final layer weights + biases
        final_x = tf.get_variable('final_x', [hidden_size, ])
        final_b = tf.get_variable('final_b', [vocab_size_word])
        # Project the decoder output to a word_vocab_size dimension
        # logits = tf.layers.dense(decoder_outputs, params['word_vocab_size'])
        # Take the arg max for each, ie, the word id.
        if mode == tf.estimator.ModeKeys.TRAIN:
            decoder_o = labels['sequence']
            print(decoder_o)
            print(decoder_outputs)
            # d_reshape = tf.reshape(decoder_outputs, [batch_size, num]) 
            loss = tf.nn.sampled_softmax_loss(weights=final_x,
                                              biases=final_b,
                                              labels=decoder_o,
                                              inputs=decoder_outputs,
                                              num_sampled=1000,
                                              num_classes=vocab_size_word,
                                              partition_strategy="div")
        else:
            logit = tf.matmul(decoder_outputs, tf.transpose(final_x))
            logit = tf.nn.bias_add(logit, final_b)
            preds = tf.cast(tf.argmax(logit, axis=2), tf.int32)
            if mode == tf.estimator.ModeKeys.PREDICT:
                # Return a dict with the sample word ids.
                preds = {"sequence": preds}
                export = {
                    'prediction': tf.estimator.export.PredictOutput(preds)
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=preds,
                                                  export_outputs=export)
            deco = labels['sequence']
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=deco,
                                                                  logits=logit)

        # Create a mask for padding characters
        target_w = tf.sequence_mask(labels['sequence_length'],
                                    dtype=logit.dtype)
        batch_size_32 = tf.cast(batch_size, tf.float32)
        timesteps_32 = tf.cast(timesteps, tf.float32)
        # Apply the mask and normalize the loss for the batch size. Could also
        # Normalize the loss for the sequence length
        loss = (tf.reduce_sum(loss * target_w) / (batch_size_32+timesteps_32))
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
