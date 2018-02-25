# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing PTB text files."""
import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile




def _read_words(filename):
    with gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()



def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = zip(*count_pairs)
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data(data_path=None):

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_iterator(raw_data, batch_size, num_steps):

    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.float32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

def generate_predict_data(raw_data, batch_size):

    raw_data = np.array(raw_data, dtype=np.int32)
    x = raw_data[0:-1].reshape([batch_size, len(raw_data) - 1])
    y = raw_data[1:len(raw_data)].reshape([batch_size, len(raw_data) - 1])
    return x, y

