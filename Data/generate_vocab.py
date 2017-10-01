# -*- coding: utf-8 -*-
"""
The goal of this file is to generate words and characters vocabularies that can
be used to encode the dataset.
To start with, all the strings will be splited according to the space char.
"""

import io
import json
import collections
from nltk.tokenize import word_tokenize

# First define the variables to hold the vocabularies. We use collections.Counter
# to keep track of the word occurences. We can then take the most X counter words
# Add a word with counter[word] += 1
# take the 10 most counted items with: Counter.most_common(10)
character_vocab = collections.Counter()
word_vocab = collections.Counter()

# Iterate over the dataset and add each word and character to their respective
# dataset.
line_counter = 0
filenames = ["data/europarl-v7.fr-en.fr", "data/news.2014.fr.shuffled.v2"]
for filename  in filenames:
    with io.open(filename, 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            line_counter += 1
            # Strip the line of new line characters \n
            line = line.strip()
            # Split the line according to the space char
            words = word_tokenize(line)
            # For every word in the line, increment the counter
            for word in words:
                word_vocab[word.lower()] += 1
            # Split the line according to characters and increment the counter
            for char in line:
                character_vocab[char] += 1
            # If this is a the X line, print it.
            if line_counter % 10000 == 0:
                print(str(line_counter))

# We now have two counters. We need to convert these to regular dict that we
# will export as JSON objects.
# For the words we will only take the most common 50.000 words
words_vocab_most = word_vocab.most_common(50000) # Returns a list [(word, freq)]
# Order the words by frequency.
words_vocab_sort = [x[0] for x in words_vocab_most]
words_vocab_sort.insert(0, '|UNK|') # Will be 3
words_vocab_sort.insert(0, '|EOS|') # Will be 2
words_vocab_sort.insert(0, '|GOO|') # Will be 1
words_vocab_sort.insert(0, '|PAD|') # Will be 0
words_vocab_dict = {word:i for i, word in enumerate(words_vocab_sort)}
words_vocab_reve = {i:word for i, word in enumerate(words_vocab_sort)}

# For characters we will take all of them.
# Order the words by frequency.
char_vocab_sort = [x[0] for x in character_vocab.items()]
char_vocab_sort.insert(0, '|UNK|') # Will be 2
char_vocab_sort.insert(0, '|PAD|') # Will be 0
char_vocab_dict = {char:i for i, char in enumerate(char_vocab_sort)}
char_vocab_reve = {i:char for i, char in enumerate(char_vocab_sort)}


with io.open("data/words_vocab_dict.json", 'w', encoding='utf8') as fin:
    fin.write(json.dumps(words_vocab_dict))

with io.open("data/words_vocab_reve.json", 'w', encoding='utf8') as fin:
    fin.write(json.dumps(words_vocab_reve))

with io.open("data/char_vocab_dict.json", 'w', encoding='utf8') as fin:
    fin.write(json.dumps(char_vocab_dict))

with io.open("data/char_vocab_reve.json", 'w', encoding='utf8') as fin:
    fin.write(json.dumps(char_vocab_reve))
