# -*- coding:utf-8 -*-
__author__ = "ChenJun"

import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf


class BatchNormLayer(object):
    """
    batch normalization.
    """
    def __init__(self, n_in, inputs):
        self.gamma = tf.Variable(initial_value=np.ones(n_in,), dtype=tf.float32, name="gamma")
        self.beta = tf.Variable(initial_value=np.zeros(n_in,), dtype=tf.float32, name="beta")
        mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
        self.out = tf.nn.batch_normalization(inputs, mean, var, offset=self.beta, scale=self.gamma, variance_epsilon=1e-6, name="bn")


class InteractLayer(object):
    """
    interact for question_vec and answer_vec.
    math: (q * W) * a
    q * W: [batch_size, n_q] * [n_q, dim, n_a] -> [batch_size, dim, n_a]
    (q * W) * a: [batch_size, dim, n_a] * [batch_size, n_a, 1] -> [batch_size, dim, 1]
    out: [batch_size, dim, 1] -> [batch_size, dim]
    """
    def __init__(self, n_q, n_a, dim):
        # self.W = tf.Variable(initial_value=tf.random_uniform(shape=(n_q, dim, n_a), minval=-0.1, maxval=0.1), name="IL_W")
        self.W = tf.Variable(tf.random_normal(shape=(n_q, dim, n_a)) * 0.05, name="IL_W")
        self.dim = dim

    def __call__(self, q_input, a_input):
        qa_vec = tf.matmul(tf.tensordot(q_input, self.W, axes=[[1], [0]]), tf.expand_dims(a_input, 2))
        out_put = qa_vec[:, :, -1]
        return out_put


class LogisticRegression(object):
    """
    logistic regression layer.
    label: one_hot [[0,1], [0,1], [1,0], ...] -> tf.nn.softmax_cross_entropy_with_logits
           number  [1, 0, 0, 1, 0, ...] -> tf.nn.sparse_softmax_cross_entropy_with_logits
    math: softmax(W * X + b)
    """
    def __init__(self, input, n_in, n_out):
        self.W = tf.Variable(tf.zeros(shape=(n_in, n_out)), name="LR_W")
        self.b = tf.Variable(tf.zeros(shape=(n_out,)), name="LR_b")
        self.linear = tf.add(tf.matmul(input, self.W), self.b)
        self.p_y_given_x = tf.nn.softmax(tf.add(tf.matmul(input, self.W), self.b))
        self.y_pred = tf.arg_max(self.p_y_given_x, 1)

    def cross_entropy(self, y):
        # cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(y, dtype=tf.float32) * tf.log(self.p_y_given_x), reduction_indices=1))  # softmax(WX+b); one-hot label
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.linear, labels=y))  # unscaled(WX+b); one-hot label
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.linear, labels=y))  # unscaled(WX+b); number label
        return cost


    def errors(self, y):
        return tf.reduce_mean(tf.cast(tf.not_equal(self.y_pred, tf.arg_max(y,1)), dtype=tf.float32))

    def pred_prob(self):
        return self.p_y_given_x[:,1]


class HiddenLayer(object):
    """
    hidden layer
    math: activation(W * X + b)
    """
    def __init__(self, input, n_in, n_out, activation=tf.tanh):
        self.input = input
        w_value = tf.random_uniform(minval=-np.sqrt(6.0 / (n_in + n_out)), maxval=np.sqrt(6.0 / (n_in + n_out)), shape=(n_in, n_out))
        if activation == tf.sigmoid:
            w_value *= 4
        self.W = tf.Variable(initial_value=w_value, name="HL_W")
        self.b = tf.Variable(initial_value=tf.zeros(shape=(n_out,)), name="HL_b")

        if activation == None:
            output = tf.add(tf.matmul(input, self.W), self.b)
        else:
            output = activation(tf.add(tf.matmul(input, self.W), self.b))
        self.output = output


class DropoutHiddenLayer(HiddenLayer):
    """
    dropout after hidden layer.
    """
    def __init__(self, input, n_in, n_out, keep_prop, activation):
        super(DropoutHiddenLayer, self).__init__(input=input, n_in=n_in, n_out=n_out, activation=activation)
        self.output = tf.nn.dropout(self.output, keep_prob=keep_prop)


class MLPDropout(object):
    """
    dropout mlp(
        hidden layer
        dropout
        logistic regression layer
        )
    """
    def __init__(self, input, n_in, n_hidden, n_out, keep_prop, activation=tf.tanh):
        self.drop_hidden_layer = DropoutHiddenLayer(input=input, n_in=n_in, n_out=n_hidden, activation=activation, keep_prop=keep_prop)
        self.logistic_regression_layer = LogisticRegression(input=self.drop_hidden_layer.output, n_in=n_hidden, n_out=n_out)

        self.cross_entropy = (
            self.logistic_regression_layer.cross_entropy
        )
        # same holds for the function computing the number of errors
        self.errors = self.logistic_regression_layer.errors
        self.pred_prob = self.logistic_regression_layer.pred_prob

        self.L1 = (
            tf.reduce_sum(abs(self.drop_hidden_layer.W)) +
            tf.reduce_sum(abs(self.logistic_regression_layer.W))
        )

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            tf.nn.l2_loss(self.drop_hidden_layer.W) +
            tf.nn.l2_loss(self.logistic_regression_layer.W)
        )


class MLP(object):
    """
    mlp(
        hidden layer
        logistic regression layer
        )
    """
    def __init__(self, input, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(input=input, n_in=n_in, n_out=n_hidden, activation=tf.tanh)
        self.logistic_regression_layer = LogisticRegression(input=self.hidden_layer.output, n_in=n_hidden, n_out=n_out)

        self.cross_entropy = (
            self.logistic_regression_layer.cross_entropy
        )
        # same holds for the function computing the number of errors
        self.errors = self.logistic_regression_layer.errors
        self.pred_prob = self.logistic_regression_layer.pred_prob

        self.L1 = (
            tf.reduce_sum(abs(self.hidden_layer.W)) +
            tf.reduce_sum(abs(self.logistic_regression_layer.W))
        )

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            tf.nn.l2_loss(self.hidden_layer.W) +
            tf.nn.l2_loss(self.logistic_regression_layer.W)
        )


