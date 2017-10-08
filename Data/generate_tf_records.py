# -*- coding: utf-8 -*-
"""
The goal of this file is to generate the training examples for the model.
"""
import io
from mistake import Mistake
import tensorflow as tf
from utils import create_example, get_vocab
import random

# Useful variables
i = 0
random.seed(0)
mistake_generator = Mistake()
c_vocab = get_vocab('data/vocab/char_vocab_dict.json')
w_vocab = get_vocab('data/vocab/words_vocab_dict.json')

# Files to write the tf example to.
training_writer = tf.python_io.TFRecordWriter("data/training.tfrecord")
validation_writer = tf.python_io.TFRecordWriter("data/validation.tfrecord")
testing_writer = tf.python_io.TFRecordWriter("data/testing.tfrecord")

# For each traning file
filenames = ["data/europarl-v7.fr-en.fr", "data/news.2014.fr.shuffled.v2"]
for filename in filenames:
    # Open the file and generate a tf.example for each line in it.
    with io.open(filename, 'r', encoding='utf8') as fin:
        # For every line in the document.
        for line in fin.readlines():
            line = line.strip()  # Strip the line of surrounding whitespaces
            sentence_length = len(line)  # Store the sequence length
            # If the sentence is too short or too long, discard it.
            if (sentence_length < 10) | (sentence_length > 750):
                continue
            # This is the correct line -> the output, what we aim to achieve
            output_seq = line.lower()  # make it lowercase
            # This is the input to the model. A sentence with errors in it.
            input_seq = mistake_generator.gen_mistake(line)
            # Make a tf.example out of the two lines
            example = create_example(input_seq, output_seq, w_vocab, c_vocab)
            # Write the tf.example to a tf.record file
            random_number = random.random()  # Random number between 0 and 1.
            # Depending on this random number, either write to the training,
            # validation or test set.
            if random_number <= 0.005:
                validation_writer.write(example.SerializeToString())
            elif random_number <= 0.01:
                testing_writer.write(example.SerializeToString())
            else:
                training_writer.write(example.SerializeToString())
            i += 1  # Track the number of lines we have been through.
            if i % 10000 == 0:
                print('Starting line number {i}'.format(i=str(i)))
                break
    break
validation_writer.close()
testing_writer.close()
training_writer.close()
