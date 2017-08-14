#-*- coding:utf-8 -*-
__author__ = "ChenJun"

import tensorflow as tf
from tensorflow.contrib import rnn
from layers import MLP, InteractLayer, BatchNormLayer, MLPDropout


class RNNModule(object):
    def __init__(self, n_hidden, cell="GRU"):
        """
        qa_rnn module init.
        :param n_hidden: num of hidden units
        :param cell: gru|lstm|basic_rnn
        """
        self.rnn_cell = rnn.BasicRNNCell(num_units=n_hidden)
        if cell == "GRU":
            self.rnn_cell = rnn.GRUCell(num_units=n_hidden)
        elif cell == "LSTM":
            self.rnn_cell = rnn.LSTMCell(num_units=n_hidden)
        else:
            raise Exception(cell + " not supported.")

    def __call__(self, q_input, a_input):
        """
        rnn layer for qa sequence.
        :param q_input: q_input, [batch, sentence_length, emb_dim]
        :param a_input: a_input, [batch, sentence_length, emb_dim]
        :return:
        """
        q_seq_outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell, inputs=q_input, dtype=tf.float32)
        tf.get_variable_scope().reuse_variables()  # reuse
        a_seq_outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell, inputs=a_input, dtype=tf.float32)
        q_sentence_vec = q_seq_outputs[:, -1, :]  # choose last state
        a_sentence_vec = a_seq_outputs[:, -1, :]  # choose last state
        return q_sentence_vec, a_sentence_vec


class QARNNModel(object):
    def __init__(self, sequence_length, n_hidden_rnn, n_in_mlp, n_hidden_mlp, n_out,
            L1_reg, L2_reg, learning_rate, word_embedding, non_static):
        """
        question-answer rnn model init and definition.
        :param sequence_length: sequence length
        :param n_hidden_rnn: rnn hidden units
        :param n_in_mlp: mlp input size
        :param n_hidden_mlp: mlp hidden size
        :param n_out: mlp out size
        :param L1_reg: mlp L1 loss
        :param L2_reg: mlp L2 loss
        :param learning_rate: learning rate for update
        :param word_embedding: word embedding
        :param non_static: bool, update embedding or not
        """
        self.lr = learning_rate
        self.word_embedding = word_embedding
        # define the placeholder
        with tf.name_scope('placeholder'):
            self.q_input = tf.placeholder(tf.int64, shape=[None, sequence_length], name='query_input')
            self.a_input = tf.placeholder(tf.int64, shape=[None, sequence_length], name='answer_input')
            self.l_input = tf.placeholder(tf.int64, shape=[None], name='label_input')  # one-hot -> [batch_size. n_out]
            self.keep_prop = tf.placeholder(tf.float32, name='keep_prop')
        # transfer input to vec with embedding.
        with tf.name_scope("embedding"):
            _word_embedding = tf.get_variable(name='word_emb', shape=self.word_embedding.shape, dtype=tf.float32,
                                              initializer=tf.constant_initializer(self.word_embedding),
                                              trainable=non_static)
            q_embedding = tf.nn.embedding_lookup(_word_embedding, self.q_input)
            a_embedding = tf.nn.embedding_lookup(_word_embedding, self.a_input)
            print "input shape(embedding): ", q_embedding.get_shape()
        # define rnn model.
        with tf.variable_scope("RNN"):
            # rnn layer
            rnn_layer = RNNModule(n_hidden_rnn, cell="GRU")
            q_sentence_vec, a_sentence_vec = rnn_layer(q_embedding, a_embedding)
        # define classifier.
        with tf.name_scope("MLPDrop"):
            interact_layer = InteractLayer(n_hidden_rnn, n_hidden_rnn, dim=n_in_mlp)
            qa_vec = interact_layer(q_sentence_vec, a_sentence_vec)
            bn_layer = BatchNormLayer(n_in=n_in_mlp, inputs=qa_vec)
            classifier = MLP(bn_layer.out, n_in_mlp, n_hidden_mlp, n_out)
            # classifier = MLPDropout(bn_layer.out, n_in_mlp, n_hidden_mlp, n_out, keep_prop=self.keep_prop)
        # define cost, optimizer and output.
        self.pred_prob = classifier.pred_prob()
        self.error = classifier.errors(self.l_input)
        self.cost = classifier.cross_entropy(self.l_input) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.9).minimize(self.cost)

    def train_batch(self, sess, qsent_batch, asent_batch, label_batch, keep_prop):
        """
        qa_rnn model with mini_batch for train.
        :param sess: tensorflow session.
        :param qsent_batch: batch of question.
        :param asent_batch: batch of answer.
        :param label_batch: batch of label
        :param keep_prop: keep prop for dropout.
        :return:
        """
        res = [self.optimizer, self.cost]
        _, cost, = sess.run(res,
                            feed_dict={self.q_input: qsent_batch,
                                       self.a_input: asent_batch,
                                       self.l_input: label_batch,
                                       self.keep_prop: keep_prop
                                       })
        return cost

    def eval_batch(self, sess, qsent_batch, asent_batch, label_batch, keep_prop):
        """
        qa_rnn model with mini_batch for eval(valid|test).
        :param sess: tensorflow session.
        :param qsent_batch: batch of question.
        :param asent_batch: batch of answer.
        :param label_batch: batch of label
        :param keep_prop: keep prop for dropout.
        :return:
        """
        res = [self.pred_prob]
        pred = sess.run(res,
                        feed_dict={self.q_input: qsent_batch,
                                   self.a_input: asent_batch,
                                   self.l_input: label_batch,
                                   self.keep_prop: keep_prop
                                   })[0]
        return pred
