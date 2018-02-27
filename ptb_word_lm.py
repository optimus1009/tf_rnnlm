# python ptb_word_lm.py --data_path=simple-examples/data/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import Reader

import myreader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", './data/simple-examples/data', "data_path")
flags.DEFINE_string("checkpoint_dir", "./corpus_logdir_medium", "checkpoint_dir")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("train", True, "should we train or test")
flags.DEFINE_string("train_dataset", "../corpus/corpus_beibei_train.data", "train data path")

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
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/gpu:0"):
            embedding = tf.get_variable( "embedding", [vocab_size, size], dtype=data_type())
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
        #logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

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
        return  self._out_probs


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 15
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
    num_steps = 30
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 30
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 50
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
    vocab_size = 150000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 100000


def run_epoch(session, model, data, eval_op, epoch_num):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(Reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        fetches = [model.cost, model.final_state, model.logits, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, state, logits, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps
        # Rani: show the actual prediction
        # decodedWordId = int(np.argmax(logits))
        # print(" ".join([inverseDictionary[int(x1)] for x1 in np.nditer(x)]) + \
        #       " got:" + inverseDictionary[decodedWordId] + " expected:" + inverseDictionary[int(y)])

        #if verbose and step % (epoch_size // 10) == 10:
        if step % 10 == 0:
            print("epoch: %d\t%.3f perplexity: %.3f speed: %.0f wps" % (epoch_num, step * 1.0 / epoch_size, np.exp(costs / iters), iters * model.batch_size / (time.time() - start_time)))
    print("costs is %d and iters is %d" % (costs,iters) )
    return np.exp(costs/iters)


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


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # raw_data = reader.ptb_raw_data(FLAGS.data_path)
    # train_data, valid_data, test_data, vocabulary, word_to_id = raw_data
    #Rani: added inverseDictionary
    word_to_id = Reader._build_vocab(FLAGS.train_dataset, cut_vocab_size=config.vocab_size)
    train_data = Reader._file_to_word_ids('../corpus/corpus_beibei_train.data',word_to_id)
    valid_data = Reader._file_to_word_ids('../corpus/corpus_beibei_valid.data',word_to_id)
    test_data  = Reader._file_to_word_ids('../corpus/corpus_beibei_test.data', word_to_id)
    # inverseDictionary = dict(zip(word_to_id.values(), word_to_id.keys()))


    #config_gpu = tf.ConfigProto()
    #config_gpu.gpu_options.allocator_type = 'BFC' #动态分配显存
    with tf.Graph().as_default(), tf.Session(config = tf.ConfigProto(log_device_placement=True)) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        if FLAGS.train:
            print('training')

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %f Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, m.train_op,i + 1)
                print("Epoch: %f Train Perplexity: %.3f" % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(), i + 1)
                print("Epoch: %f Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                saver.save(session, FLAGS.checkpoint_dir + '/model.ckpt', global_step=i + 1)
        else:
            print('testing')
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint file found")

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
    # print (FLAGS.checkpoint_dir)
