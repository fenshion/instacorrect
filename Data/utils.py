# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 08:55:51 2017

@author: maxime
"""

import tensorflow as tf
import io
import json
from nltk.tokenize import word_tokenize

def encode_line(line, vocab):
    """Given a string and a vocab dict, encodes the given string"""
    line = line.strip()
    sequence = [vocab.get(char, vocab['<UNK>']) for char in line]
    sequence_length = len(sequence)
    return sequence, sequence_length

def encode_line_charwise(line, vocab):
    """Given a string will encode it into the right tf.example format"""
    # Encode the string into their vocab representation
    splited = word_tokenize(line)
    sequence_length = len(splited)
    max_word_length = max([len(word) for word in splited])
    # Should have a one array of int.
    sequence = []
    pad_char = vocab.get('|PAD|')
    for word in splited:
        word_encoded = [vocab.get(char, vocab['|UNK|']) for char in word]
        word_encoded += [pad_char]*(max_word_length - len(word))
        sequence.extend(word_encoded)
    return sequence, sequence_length, max_word_length

def encode_line_wordwise(line, vocab):
    """Given a string and vocab, return the word encoded version"""
    splited = word_tokenize(line)
    sequence_input = [vocab.get(word, vocab['|UNK|']) for word in splited]
    sequence_input = [vocab['|GOO|']] + sequence_input
    sequence_output = sequence_input[1:] + [vocab['|EOS|']]
    sequence_length = len(sequence_input)
    return sequence_input, sequence_output, sequence_length

def encode_line_wordwise_transformer(line, vocab):
    """Given a string and vocab, return the word encoded version"""
    splited = word_tokenize(line)
    sequence_input = [vocab.get(word, vocab['|UNK|']) for word in splited]
    sequence_output = sequence_input + [vocab['|EOS|']]
    sequence_length = len(sequence_output)
    return sequence_output, sequence_length

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(inputs, outputs, word_vocab, char_vocab):
    """Given a string and a label (and a vocab dict), returns a tf.Example"""
    inp_seq, inp_sl, inp_maxword = encode_line_charwise(inputs, char_vocab)
    out_seq, out_sl1, out_maxword = encode_line_charwise(outputs, char_vocab)
    out_seq_w, out_sl2 = encode_line_wordwise_transformer(outputs, word_vocab)
    example = tf.train.Example(features=tf.train.Features(feature={
            'input_sequence': _int64_feature(inp_seq),
            'input_sequence_length': _int64_feature([inp_sl]),
            'input_sequence_maxword': _int64_feature([inp_maxword]),
            'output_sequence': _int64_feature(out_seq),
            'output_sequence_words': _int64_feature(out_seq_w),
            'output_sequence_length': _int64_feature([out_sl1]),
            'output_sequence_maxword':_int64_feature([out_maxword])}))
    return example

def get_vocab(filename):
    with io.open(filename, 'r', encoding='utf8') as fin:   
        vocab=json.loads(fin.readline())
    return vocab