import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import os
import json
from input_functions import input_fn, serving_input_receiver_fn
from trans_model import transformer
import argparse
import io
from tensorflow.python import debug as tf_debug
# hooks = [tf_debug.LocalCLIDebugHook()]
hooks = []


def get_vocab(filename):
    with io.open(filename, 'r', encoding='utf8') as fin:
        vocab = json.load(fin)
    return vocab

char_vocab = get_vocab('../Data/data/vocab/char_vocab_dict.json')
word_vocab = get_vocab('../Data/data/bpe/apply_bpe.txt.json')
reve_vocab = get_vocab('../Data/data/bpe/apply_bpe.txt_reve.json')
# Parameters given to the estimator. Mainly the size of the vocabulary
# The batch size to use for the train/valid/test set
batch = 12
# the embedding size to use and the (keep) drop out percentage
model_params = {'char_vocab_size': len(char_vocab),
                'word_vocab_size': len(word_vocab),
                'char_embedding_size': 15,
                'word_embedding_size': 125,
                'dropout': 0.2,
                'hidden_size': 512,
                'learning_rate': 0.00001,
                'decay_steps': 100000,
                'kernels': [2, 3, 4, 5, 6],
                'kernel_features': [50, 100, 150, 200, 200],
                'ultimate_sequ_len': 100,
                'num_blocks': 3,
                'attention_heads': 8,
                'go_id': word_vocab['GOO'],
                'eos_id': word_vocab['EOS']}
# The number of times to train the model on the entire dataset
epochs = 100000
# The part of the dataset that will be skipped to be used by the training
# and testing dataset
# Lambda function used in the experiment. Returns a dataset iterator
data_train = lambda: input_fn("../Data/data/training.tfrecord", batch, epochs)
data_valid = lambda: input_fn("../Data/data/validation.tfrecord", batch, 1)

# Set the TF_CONFIG environment to local to avoid bugs
os.environ['TF_CONFIG'] = json.dumps({'environment': 'local'})


def train():
    """Perform the training of the model"""
    # Create a run config
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=60*60,
                                        log_device_placement=True,
                                        tf_random_seed=0,
                                        save_summary_steps=1000)
    # Create the estimator, the actual RNN model, with the defined directory
    # for storing files, the parameters, and the config.
    estimator = tf.estimator.Estimator(model_fn=transformer,
                                       model_dir="output/",
                                       params=model_params,
                                       config=config)
    # Give this estimator to an experiment to run the traning and validation,
    # input_functions, etc. From the tf documentation: "After an experiment is
    # created (by passing an Estimator and inputs for training and evaluation),
    # an Experiment instance knows how to invoke training and eval loops [...]
    # eval_step = None so the evaluation step uses the entire validation set
    experiment = tf.contrib.learn.Experiment(estimator=estimator,
                                             train_input_fn=data_train,
                                             eval_input_fn=data_valid,
                                             eval_steps=None,
                                             local_eval_frequency=1,
                                             min_eval_frequency=1,
                                             train_monitors=hooks,
                                             eval_hooks=hooks)
    experiment.train_and_evaluate()


def inference(rng):
    """Perform an inference on the latest checkpoint available"""
    # Lambda function used in the experiment. Returns a dataset iterator
    data_test = lambda: input_fn("../Data/data/validation.tfrecord", 5, 1,
                                 take=10)
    # Create a run config
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=60*60,
                                        log_device_placement=True,
                                        tf_random_seed=0,
                                        save_summary_steps=1000)
    # Create the estimator, the actual RNN model, with the defined directory
    # for storing files, the parameters, and the config.
    estimator = tf.estimator.Estimator(model_fn=transformer,
                                       model_dir="output/",
                                       params=model_params,
                                       config=config)
    result = estimator.predict(data_test)
    print(result)
    sequence = next(result)
    print(sequence)
    while sequence:
        print(sequence)
        txt = " ".join([reve_vocab[str(wid)] for wid in sequence['sequence']])
        print(txt)
        sequence = next(result)
    return result


def export():
    """Export the last saved graph"""
    # Create a run config
    model_params['dropout'] = 0

    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=60*30,
                                        log_device_placement=True,
                                        tf_random_seed=0,
                                        save_summary_steps=1000)

    estimator = tf.estimator.Estimator(model_fn=transformer,
                                       model_dir="output/",
                                       params=model_params,
                                       config=config)
    estimator.export_savedmodel("output/model_serving",
                                serving_input_receiver_fn)


def input_inspection():
    """Inspect the inputs for inconsistency"""

    reverse_char_vocab = get_vocab('../Data/data/vocab/char_vocab_reve.json')
    char_vocab = get_vocab('../Data/data/vocab/char_vocab_dict.json')
    reverse_word_vocab = get_vocab('../Data/data/bpe/apply_bpe.txt_reve.json')

    f, l = data_train()

    sess = tf.Session()
    features, labels = sess.run([f, l])
    inputs = features['sequence']
    output = labels['sequence']


    pad_char = char_vocab['|PAD|']
    for i in range(inputs.shape[0]):
        shape = inputs[i, :, :].shape
        sentence = inputs[i, :, :].tolist()
        for x in range(shape[0]):  # Number of words
            for y in range(shape[1]):  # Words length
                if sentence[x][y] != pad_char:
                    print(reverse_char_vocab.get(str(sentence[x][y]), '<UNK>'),
                          end='')
            print(' ', end='')
        print('END')
        print('****')

    for i in range(output.shape[0]):
        ids = output[i, :].tolist()
        words = [reverse_word_vocab.get(str(wid)) for wid in ids]
        print(" ".join(words))


if __name__ == "__main__":

    # hey=inference(['Maxime'])
    # next(hey)

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', dest='predict', action="store_true")
    parser.add_argument('--export', dest='export', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--inspection', dest='inspection', action='store_true')
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--to_predict', type=str, help='The str to pred.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.predict:
        inference(5)
    elif FLAGS.export:
        print('Start export')
        export()
    elif FLAGS.inspection:
        input_inspection()
    elif FLAGS.train:
        train()
    else:
        pass
