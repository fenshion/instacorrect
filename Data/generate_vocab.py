# -*- coding: utf-8 -*-
"""
The goal of this file is to generate words and characters vocabularies that can
be used to encode the dataset.
To start with, all the strings will be splited according to the space char.
"""

import io
import json
import collections

# First define the variables to hold the vocabularies. We use
# collections.Counter to keep track of the word occurences.
# We can then take the most X counter words, add a word with counter[word] += 1
# take the 10 most counted items with: Counter.most_common(10)
character_vocab = collections.Counter()

# Iterate over the dataset and add each word and character to their respective
# dataset.
line_counter = 0
filenames = ["data/europarl-v7.fr-en.fr", "data/news.2014.fr.shuffled.v2"]
for filename in filenames:
    with io.open(filename, 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            line_counter += 1
            # Strip the line of new line characters \n
            for char in line:
                character_vocab[char] += 1
            # If this is a the X line, print it.
            if line_counter % 10000 == 0:
                print(str(line_counter))

# For characters we will take all of them.
# Order the words by frequency. -> small numbers appear more often, max space
char_vocab_sort = [x[0] for x in character_vocab.most_common()]
char_vocab_sort.insert(0, '}')  # Will be 0
char_vocab_sort.insert(0, '{')  # Will be 0
char_vocab_sort.insert(0, '|')  # Will be 0
char_vocab_sort.insert(0, '')  # Will be 0
char_vocab_sort.insert(0, '|GOO|')  # Will be 0
char_vocab_sort.insert(0, '|EOS|')  # Will be 0
char_vocab_sort.insert(0, '|UNK|')  # Will be 2
char_vocab_sort.insert(0, '|PAD|')  # Will be 0
char_vocab_dict = {char: i for i, char in enumerate(char_vocab_sort)}
char_vocab_reve = {i: char for i, char in enumerate(char_vocab_sort)}


filename = "data/vocab/char_vocab_reve.json"
with open(filename, 'w', encoding="utf8") as f:
    json.dump(char_vocab_reve, f, indent=2, ensure_ascii=False)

filename = "data/vocab/char_vocab_dict.json"
with open(filename, 'w', encoding="utf8") as f:
    json.dump(char_vocab_dict, f, indent=2, ensure_ascii=False)
