# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:55:36 2017

@author: maxime
"""

import tf_glove
import io
from nltk.tokenize import word_tokenize

path = "data/europarl-v7.fr-en.fr"

def extract_corpus(path):
    """Returns a generator that reads a line at a time"""
    with io.open(path, 'r', encoding='utf8') as file:
        for line in file:
            yield line.lower()
            
def generate_corpus(path):
    """Returns a generator that loops over the text"""
    return (word_tokenize(line) for line in extract_corpus(path))

corpus = generate_corpus(path)

model = tf_glove.GloVeModel(context_size=10, batch_size=254, min_occurrences=25,
                            learning_rate=0.05, embedding_size=50,
                            max_vocab_size=100000)

model.fit_to_corpus(corpus)
model.train(num_epochs=50, log_dir="log/example", summary_batch_interval=1000)

model.embedding_for("maxime")
