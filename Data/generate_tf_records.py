# -*- coding: utf-8 -*-
"""
The goal of this file is to generate the training examples for the model.
"""
import io
from mistake import Mistake
import tensorflow as tf
import random
from nltk.tokenize import word_tokenize
import json


def get_vocab(filename, multiple_lines=False):
    with io.open(filename, 'r', encoding='utf8') as fin:
        vocab = json.load(fin)
    return vocab

# Useful variables
i = 0
random.seed(0)
mistake_generator = Mistake()
c_vocab = get_vocab('data/vocab/char_vocab_dict.json')
# w_vocab = get_vocab('data/vocab/words_vocab_dict.json')
bpe_vocab = get_vocab('data/bpe/apply_bpe.txt.json')

# Files to write the tf example to.
training_writer = tf.python_io.TFRecordWriter("data/training.tfrecord")
validation_writer = tf.python_io.TFRecordWriter("data/validation.tfrecord")
testing_writer = tf.python_io.TFRecordWriter("data/testing.tfrecord")


def encode_line_charwise(line, vocab):
    """Given a string will encode it into the right tf.example format"""
    # Encode the string into their vocab representation
    splited = word_tokenize(line)
    sequence_length = len(splited)
    max_word_length = max([len(word)+2 for word in splited])
    # Should have a one array of int.
    sequence = []
    pad_char = vocab.get('|PAD|')
    for word in splited:
        word_encoded = [vocab.get('{')] + [vocab.get(char, vocab['|UNK|'])
                                           for char in word] + [vocab.get('}')]
        word_encoded += [pad_char]*(max_word_length - len(word_encoded))
        sequence.extend(word_encoded)
    return sequence, sequence_length, max_word_length


def encode_line_wordwise(line, vocab):
    """Given a string and vocab, return the word encoded version"""
    splited = line.split()
    sequence_input = [vocab.get(word, vocab['UNK']) for word in splited]
    sequence_input = [vocab['GOO']] + sequence_input + [vocab['EOS']]
    sequence_length = len(sequence_input) - 1
    return sequence_input, sequence_length


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# For each line in dataset and bpe encoded dataset
filenames = ['data/dataset.txt', 'data/bpe/apply_bpe.txt']
with io.open(filenames[0], 'r', encoding='utf8') as textfile1:
    with io.open(filenames[1], 'r', encoding='utf8') as textfile2:
        for inp_s, out_s in zip(textfile1, textfile2):
            inp_s = inp_s.strip().lower()
            if (len(inp_s) < 10) | (len(inp_s) > 500):
                continue
            inp_s = mistake_generator.gen_mistake(inp_s)
            out_s = out_s.strip()
            inp_seq, inp_seq_len, inp_max = encode_line_charwise(inp_s,
                                                                 c_vocab)
            out_seq, out_seq_len = encode_line_wordwise(out_s, bpe_vocab)

            example = tf.train.Example(features=tf.train.Features(feature={
                'input_sequence': _int64_feature(inp_seq),
                'input_sequence_length': _int64_feature([inp_seq_len]),
                'input_sequence_maxword': _int64_feature([inp_max]),
                'output_sequence': _int64_feature(out_seq),
                'output_sequence_length': _int64_feature([out_seq_len])}))

            random_number = random.random()  # Random number between 0 and 1.
            if random_number <= 0.005:
                validation_writer.write(example.SerializeToString())
            elif random_number <= 0.01:
                testing_writer.write(example.SerializeToString())
            else:
                training_writer.write(example.SerializeToString())
            i += 1  # Track the number of lines we have been through.
            if i % 1000 == 0:
                print('Starting line number {i}'.format(i=str(i)))

validation_writer.close()
testing_writer.close()
training_writer.close()
