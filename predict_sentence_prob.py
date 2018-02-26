from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import random

import myreader as reader
import Reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", './data/simple-examples/data', "data_path")
flags.DEFINE_string("checkpoint_dir", "../corpus_logdir_larger_vocab", "checkpoint_dir")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_string("train_dataset", "../corpus/train_corpus_wiki.data", "train data path")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        # logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #     [logits],
        #     [tf.reshape(self._targets, [-1])],
        #     [tf.ones([batch_size * num_steps], dtype=data_type())])
        logits_ = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        loss = tf.contrib.seq2seq.sequence_loss(
            logits_,
            self.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = state
        # RANI
        self.logits = logits_

        if not is_training:
            self._out_probs = tf.nn.softmax(logits_)
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def out_probs(self):
        return self._out_probs


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 20
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 100000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 200
    vocab_size = 100000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 200
    vocab_size = 100000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 20
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 100000


def run_epoch(session, model, data, eval_op, verbose=True):
    """Runs the model on the given data."""
    state = session.run(model.initial_state)
    x, y = reader.generate_predict_data(data, batch_size=model.batch_size)
    fetches = [model.out_probs]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
    out_probs = session.run(fetches, feed_dict)

    sent_prob = 0
    for index, id_y in enumerate(y[0]):
        word_prob = out_probs[0][0:1][0][index][id_y]
        sent_prob += word_prob


    return  sent_prob/model.num_steps# + random.uniform(0.01, 0.0345)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


# Rani: save the session
def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    # raw_data = reader.ptb_raw_data(FLAGS.data_path)
    # train_data, valid_data, test_data, vocabulary, word_to_id = raw_data
    # inverseDictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    config = get_config()
    eval_config = get_config()


    word_to_id = Reader._build_vocab(FLAGS.train_dataset, cut_vocab_size=config.vocab_size)
    # predict_data = Reader._file_to_word_ids_predict('./corpus/predict_corpus_wiki.data',word_to_id)
    eval_config.batch_size = 1
    # eval_config.num_steps = len(predict_data) -1


    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        # with tf.variable_scope("model", reuse=True, initializer=initializer):
        #     mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())

        print('testing')
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")

        with open('res.data','w') as res_f:
            k = 0
            for predict_data, num_step, unk_words , unk_words_percent in Reader._file_to_word_ids_predict("../corpus/predict_corpus_wiki_seg.data", word_to_id=word_to_id):
                eval_config.num_steps = num_step
                start_time = time.time()
                with tf.variable_scope("model", reuse=True, initializer=initializer):
                    mtest = PTBModel(is_training=False, config=eval_config)
                sentence_prob = run_epoch(session, mtest, predict_data, tf.no_op())
                k += 1
                print("%d \t sentence prob: %f unknow words: %d unknow words percent: %f  cost: %f" % (k, sentence_prob, unk_words,unk_words_percent,time.time() - start_time))
                #res_f.write(str(sentence_prob) + '\n')


if __name__ == "__main__":
    tf.app.run()