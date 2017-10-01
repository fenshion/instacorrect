# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:40:12 2017

@author: maxime
"""
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import AttentionWrapper, TrainingHelper, BasicDecoder, dynamic_decode, LuongAttention, GreedyEmbeddingHelper

def create_cell(mode, dropout, num_units):
    cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
    if mode != tf.estimator.ModeKeys.PREDICT:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.8)
    return cell

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    # cnn_inputs, kernel_feature_size, 1, kernel_size
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def cnnlstm(features, labels, mode, params):
    """
    Model to be used in the tf.estimator. Basically the machine learning model.
    Simple RNN model that:
        - Takes a sentence represented like ['This', 'is', 'a', 'sentence']
          where each character in a word is represented by a integer and each word
          in a batch has the same length (zero padded)
        - One word at a time, each word is embedded using a CNN and a Highway 
          network. (TODO: add the highway network)
        - This embedding is given to a RNN
        - The last state is given to another RNN (+ Attention over the previous
          hidden state) that predicts the next word.
    
    Args:
        - features: a dict: 
            - sequence: a tensor of shape [batch_size, max_sentence_length, max_word_size]
            filled with the character ids, and padded with 0
            - sequence_length: a tensor of shape [batch_size] with the original length of 
            the sequences.
            - max_word_size: tha maximum length of each word in the batch
        - labels: a dict:
            - sequence: a tensor of shape [batch_size, max_sentence_length] filled
            with the words ids of each sentence and padded with 0.
            - sequence_length: a tensor of shape [batch_size] with the original length of 
            the sequences.
        - mode: the mode of the model (given by the estimator)
        - params: a dict with the following keys:
            - vocab_size: the size of the character vocabulary used
            - embedding_size: the size of the embeddings
            - dropout: 1 - dropout probability (the keep probability)
    """
    with tf.variable_scope('ModelParams'):
        batch_size = tf.shape(features['sequence'])[0]
        timesteps = tf.shape(features['sequence'])[1]
        maxwordlength = tf.shape(features['sequence'])[2]
        c_embed_s = params['char_embedding_size']
        dropout = params['dropout']
        hidden_size = params['hidden_size']
        network_depth = params['network_depth']
        kernels = params['kernels']
        kernel_features = params['kernel_features']
        
    with tf.variable_scope('Convolution'):
        ###########
        # ENCODER #
        ###########
        # Characters embeddings matrix. Basically each character id (int)
        # is associated a vector [char_embedding_size]
        embeddings_c = tf.Variable(tf.random_uniform([params['char_vocab_size'], 
                                   c_embed_s], -1.0, 1.0))
        # Embed every char id into their embedding. Will go from this dimension
        # [batch_size, max_sequence_length, max_word_size] to this dimension
        # [batch_size, max_sequence_length, max_word_size, char_embedding_size]
        embedded_chars = tf.nn.embedding_lookup(embeddings_c, features['sequence'])
        # Change the dimension and bring every word as an example.
        
        # Reshape the inputs to have words as second and third dimension
        # from [batch, timesteps, wordlength, embedsize] to 
        # [batch*timesteps, wordlength, embedisze]
        cnn_inputs = tf.reshape(embedded_chars, [batch_size*timesteps, 
                                                 maxwordlength, 
                                                 c_embed_s])
        # Expand the second dimension for convolution purposes
        cnn_inputs = tf.expand_dims(cnn_inputs, 1)
        # Layer to hold all of the convolution results
        layers = []
        # For each kernel, tuple of [kernel size, num filters]
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            # Apply the convolution on all of the inputs for this kernal
            conv = conv2d(cnn_inputs, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
            pool = tf.reduce_max(tf.tanh(conv), 2, keep_dims=True)
            layers.append(tf.squeeze(pool, [1, 2]))
        cnn_output = tf.concat(layers, 1)
        rnn_inputs = tf.reshape(cnn_output, [batch_size, timesteps, sum(kernel_features)])
    
    with tf.variable_scope('RNN_Encoder'):
        # Create the actual encoder. Which applies a convolution on the char input
        # to have an embedding for each word. This embedding is then fed to the 
        # classical LSMT RNN.
        # TODO: apply dropout
        cell_list = [create_cell(mode, dropout, hidden_size) for _ in range(network_depth)]
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        # Loop over the inputs and apply the previously created cell at every 
        # timestep. Returns the output at every step and last hidden state.
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, 
                                                 inputs=rnn_inputs, 
                                                 sequence_length=features['sequence_length'])
    
    with tf.variable_scope('Decoder'):
        ###########
        # DECODER #
        ###########
        # Words embeddings matrix. Basically every word id (int) in the vocab
        # is associated a vector [char_embedding_size]
        embeddings_w = tf.Variable(tf.random_uniform([params['word_vocab_size'], 
                                   params['word_embedding_size']], -1.0, 1.0))
        # Decoder cell. Basic LSTM cell that will do the decoding.
        cell_list_dec = [create_cell(mode, dropout, hidden_size) for _ in range(network_depth)]
        decoder_cell = cell = tf.contrib.rnn.MultiRNNCell(cell_list_dec)
        # Attention mechanism
        attention_mechanism = LuongAttention(num_units=hidden_size, 
                                    memory=encoder_outputs,
                                    memory_sequence_length=features['sequence_length'])
        # Attention Wrapper
        attn_cell = AttentionWrapper(decoder_cell, attention_mechanism)
        initial_decoder_state = attn_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        # Projection layer. Layer that takes the output of the decoder cell 
        # and projects it on the word vocab dimension.
        projection_layer = Dense(params['word_vocab_size'], use_bias=False)
        # If not at infering mode, use the decoder_inputs
        # output at each time step.
        if mode != tf.estimator.ModeKeys.PREDICT:
            # Decoder outputs, i.e., what we are trying to predict.
            decoder_o = tf.cast(labels['sequence_output'], tf.int32)
            # Embed the decoder input
            decoder_i = tf.nn.embedding_lookup(embeddings_w, labels['sequence_input'])
            # Helper method. Basically a function that "helps" the decoder
            # at each time step by giving it the true input, whatever it computed
            # earlier.
            output_sequence_length = tf.cast(labels['sequence_length'], tf.int32)
            helper = TrainingHelper(decoder_i, output_sequence_length)
        else:
            # Helper method. At inference time it is different, we do not have the
            # true inputs, so this function will take the previously generated output
            # and embbed it with the decoder embeddings.
            start_token = tf.fill([batch_size], params['start_token'])
            end_token = tf.cast(params['end_token'], tf.int32)
            helper = GreedyEmbeddingHelper(embeddings_w, start_token, end_token)
        # The final decoder, with its cell, its intial state, its helper function,
        # and its projection layer. 
        decoder = BasicDecoder(attn_cell, helper,initial_decoder_state, output_layer=projection_layer)
        # Use this decoder to perform a dynamic decode.
        # Dynamic Decoder: controls the flow of operations and mainly store the outputs
        # and keeps decoding until the decoder is done.
        # Decoder: kind of the cell of the dynacmic decoder. It passes the inputs
        # to the RNN, samples the output of the RNN and computes the next input.
        # To sample and compute the next inputs, the decoder uses a Helper function.
        # During training it is a TrainingHelper and during inference it is GreedyEmbeddingHelper
        # In our case the sampling is simply taking the argmax of the output logit.
        # The main difference between the two helpers is on the way they "compute" 
        # the next input. TrainingHelper will use the decoder inputs provided while 
        # the GreedyEmbeddingHelper will use the sampled RNN output and give it to
        # an embedding function to give it at as the next input. 
        # Outputs of the BasicDecoder is a BasicDecoderOutput which holds the logits
        # and the sample_ids.
        if mode != tf.estimator.ModeKeys.PREDICT:
            outputs, state, sequence_lengths = dynamic_decode(decoder)
        else:
            max_iterations = tf.cast(tf.reduce_max(features['sequence_length'])*2, tf.int32)
            outputs, state, sequence_lengths = dynamic_decode(decoder,
                                                    maximum_iterations=max_iterations)   
    with tf.variable_scope('Prediction'):
        # Contains the 
        logits = outputs.rnn_output # output of the projection layer
        sample_id = outputs.sample_id # argmax of the logits
        # If we are INFER time only
        if mode == tf.estimator.ModeKeys.PREDICT: 
            # Return a dict with the sample word ids.
            predictions = {"sequence": sample_id}
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, 
                                              export_outputs=export_outputs)
    with tf.variable_scope('Loss'):
        # We are not at INFER time. We compute the cross entropy.
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_o, 
                                                                  logits=logits)
        # Here we create a mask to "erase" the loss where the sentences are finished
        target_w = tf.sequence_mask(labels['sequence_length'], dtype=logits.dtype)
        # We apply the mask and sum the loss accross all the dimensions and divide it
        # by the batch size to make it independent of the batch_size.
        batch_size_32 = tf.cast(batch_size, tf.float32)
        timesteps_32 = tf.cast(timesteps, tf.float32)
        loss = (tf.reduce_sum(crossent * target_w) / (batch_size_32+timesteps_32))
    with tf.variable_scope('Train'):    
        # At train time only.
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Initialize an optimize that has for goal to minimize the loss
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
    with tf.variable_scope('Evaluate'):    
        # Compute the accuracy of the model (the number of sequences that the model
        # got right)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=decoder_o, 
                                                           predictions=sample_id,
                                                           weights=target_w)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)