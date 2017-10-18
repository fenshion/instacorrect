#!/usr/bin/env python

import numpy
import json

import sys
import fileinput

from collections import OrderedDict


def main():
    for filename in sys.argv[1:]:
        print('Processing', filename)
        word_freqs = OrderedDict()
        with open(filename, 'r', encoding="utf8") as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['PAD'] = 0
        worddict['EOS'] = 1
        worddict['UNK'] = 2
        worddict['GOO'] = 3
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+4
        reverse_dict = {v: k for k, v in worddict.items()}

        with open('%s.json' % filename, 'w', encoding="utf8") as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        with open('%s_reve.json' % filename, 'w', encoding="utf8") as f:
            json.dump(reverse_dict, f, indent=2, ensure_ascii=False)

        print('Done')

if __name__ == '__main__':
    main()
