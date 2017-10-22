from __future__ import print_function
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
import tensorflow as tf
import io
import json

# Load the character vocabulary
with io.open("/app/data/vocab/char_vocab_dict.json",
             'r', encoding='utf8') as fin:
        char_vocab = json.loads(fin.readline())

# Load the word vocabulary
with io.open("/app/data/vocab/words_vocab_dict.json",
             'r', encoding='utf8') as fin:
        word_vocab = json.loads(fin.readline())

# Load the rverse word vocabulary
with io.open("/app/data/vocab/words_vocab_reve.json",
             'r', encoding='utf8') as fin:
        reve_word_vocab = json.loads(fin.readline())

detokenizer = MosesDetokenizer()
eos = word_vocab['EOS']
vocab_set = set(word_vocab.keys())


def correct_sentence(sentence):
    """Ask the server to correct this sentence"""
    host = "tfserver"
    port = 9000
    model_name = "instacorrect"
    example = create_example(sentence, char_vocab).SerializeToString()
    serialized_examples = tf.contrib.util.make_tensor_proto([example])
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs['examples'].CopyFrom(serialized_examples)
    result = stub.Predict(request, 5.0)  # 5 seconds
    corrected_sentence = result.outputs['sequence'].int_val
    decoded_sentence = decode_sentence(corrected_sentence)
    return decoded_sentence


def decode_sentence(sample_ids):
    decoded = [reve_word_vocab[str(ids)] for ids in sample_ids if ids != eos]
    decoded = detokenizer.detokenize(decoded, return_str=True)
    return decoded.capitalize()


def create_example(line, vocab):
    """Given a string and a label (and a vocab dict), returns a tf.Example"""
    seq, seq_l, max_word = encode_line_charwise(line, vocab)
    features = {'input_sequence': _int64_feature(seq),
                'input_sequence_length': _int64_feature([seq_l]),
                'input_sequence_maxword': _int64_feature([max_word])}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
