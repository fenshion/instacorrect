# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 08:58:44 2017

@author: maxime

Simple Flask app that serves two purposes:
- render the index.html
- take request for corrections, send them to the tf server and give the
  result back to the browser.
"""
from __future__ import print_function
from flask import Flask, request, jsonify, render_template
import nltk
nltk.download('punkt')
nltk.download('perluniprops')
from nltk.tokenize import sent_tokenize
from correct import correct_sentence

app = Flask(__name__)


@app.route('/')
def index():
    """
    Index page of the front-end. Nothing particular
    """
    return render_template('index.html')


# curl -H "Content-Type: application/json" -X POST -d
# '{"sentence":"this is my sentence"}' http://0.0.0.0/api/is_correct
@app.route('/api/is_correct', methods=['POST'])
def is_correct():
    """
    API endpoint to correct a given sentence. Will first encode the sentence
    into then send it to the model, it will then compare the results.
    """
    data = request.get_json()
    # Check that the data received contains a sentence
    if 'sentence' not in data:
        return "Please give a sentence to correct", 400
    text = u""+data['sentence']  # Just to be sure it is a unicode string.
    # Cut the text into (possible) multiple sentences.
    sentences = sent_tokenize(text)
    if len(sentences) > 10 | len(text) > 1000:
        return "Please reduce the size of the text", 400
    corrected_sentences = []
    for i, sentence in enumerate(sentences):
        tmp_response = correct_sentence(sentence)
        corrected_sentences.append(tmp_response)
    contacted_sentences = " ".join(corrected_sentences)
    return jsonify({"sentence": contacted_sentences})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
