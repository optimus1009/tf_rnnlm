# -*- coding: utf-8 -*-
# Created by lin.xiong

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf
import logging


program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format = "%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(level=logging.INFO)

Py3 = sys.version_info[0] == 3

def _read_words(filename,delta = 1000):
    delta = delta
    with open(filename,'r',encoding="utf-8") as f:
        file_arr = f.readlines()
        num_line = len(file_arr)
        read_batch_len = num_line//delta
        s_offset = 0
        e_offset = delta
        for epoch in range(read_batch_len):
            lines = ' '.join(file_arr[s_offset:e_offset]).replace("\n", " <eos> ").split()
            s_offset += delta
            e_offset += delta
            yield lines
            logger.info(str((epoch + 1)*delta) + " lines data read")

def _build_vocab(filename, cut_vocab_size):

    logger.info("Starting building word dict")
    tmp_vocab = []
    for data in _read_words(filename):
        tmp_vocab.extend(data)
    counter = collections.Counter(tmp_vocab)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0:cut_vocab_size - 1]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    word_to_id['<unk>'] = count_pairs[-1][-1] + 1
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    logger.info("start convert file to ids")
    data = []
    for lines in _read_words(filename):
        data.extend(lines)
    #return [word_to_id[word] for word in data ]
    ids= []
    for word in data:
        if word in word_to_id:
            ids.append(word_to_id[word])
        else:
            ids.append(word_to_id["<unk>"])
    return ids

def  _file_to_word_ids_predict(filename, word_to_id):
    with open(filename,'r', encoding='utf-8') as f:
        for line in f:
            data = line.replace('\n', " <eos>").split()
            tmp_arr = []
            unk_words = 0
            for word in data:
                if word in word_to_id:
                    tmp_arr.append(word_to_id[word])
                else:
                    tmp_arr.append(word_to_id["<unk>"])
                    unk_words += 1
            predict_data = tmp_arr
            num_step = len(tmp_arr) - 1
            unk_words_percent = unk_words/ len(data)
            yield predict_data, num_step, unk_words, unk_words_percent


def ptb_raw_data(data_path=None, word_to_id = None):

    train_path = os.path.join(data_path, "demo.corpus.train.data")
    valid_path = os.path.join(data_path, "demo.corpus.valid.data")
    test_path = os.path.join(data_path, "demo.corpus.test.data")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

def produce_train_raw_data(data_path = None, word_to_id = None):
    train_path = os.path.join(data_path,"corpus.train.data")
    delta = 300000
    with open(data_path,'r',encoding="utf-8") as f:
        file_arr = f.readlines()
        num_line = len(file_arr)
        read_batch_len = num_line//delta
        s_offset = 0
        e_offset = delta
        for epoch in range(read_batch_len):
            lines = ' '.join(file_arr[s_offset:e_offset]).replace("\n", " <eos>").split()
            s_offset += delta
            e_offset += delta
            yield [word_to_id[word] for word in lines if word in word_to_id]


def ptb_producer(raw_data, batch_size, num_steps, name=None):

    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):

        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

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


if __name__ == '__main__':

    a = _build_vocab('./corpus/demo.corpus.train.data', -1)
    print(len(a))



