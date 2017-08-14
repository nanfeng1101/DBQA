#-*- coding:utf-8 -*-
__author__ = "ChenJun"

import tensorflow as tf
from layers import MLP, InteractLayer, BatchNormLayer, MLPDropout


class CNNModule(object):
    def __init__(self, feature_maps, filter_shape, pool_size):
        """
        qa_cnn module init.
        :param feature_maps: feature maps(filter_num)
        :param filter_shape: convolution filter_shape, [filter_height, filter_width, in_channels, out_channels]
        :param pool_size: max pool size, [1, img_h - filter_h + 1, img_w - filter_w + 1, 1]
        """
        self.feature_maps = feature_maps
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.W = tf.Variable(tf.truncated_normal(self.filter_shape, stddev=0.1), name="W")
        self.b = tf.Variable(tf.constant(0.1, shape=[self.feature_maps]), name="b")

    def __call__(self, q_input, a_input):
        """
        convolution + max_pooling layer.
        :param q_input: q_sentence_vec, [batch, in_height, in_width, in_channels]
        :param a_input: a_sentence_vec, [batch, in_height, in_width, in_channels]
        :return:
        """
        q_conv = tf.nn.conv2d(
            input=q_input,
            filter=self.W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="q_conv")
        # apply non_linearity
        q_h = tf.nn.relu(tf.nn.bias_add(q_conv, self.b), name="relu")
        # max_pooling over the outputs
        q_pooled = tf.nn.max_pool(
            value=q_h,
            ksize=self.pool_size,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="q_pool")

        tf.get_variable_scope().reuse_variables()  # reuse

        a_conv = tf.nn.conv2d(
            input=a_input,
            filter=self.W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="a_conv")
        # apply non_linearity
        a_h = tf.nn.relu(tf.nn.bias_add(a_conv, self.b), name="relu")
        a_pooled = tf.nn.max_pool(
            value=a_h,
            ksize=self.pool_size,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="a_pool")
        return q_pooled, a_pooled


class InceptionModule(object):
    """
    simple inception module.
    """
    def __init__(self, img_h, img_w, filter_windows, filter_num):
        """
        inception module init.
        :param img_h: sentence length
        :param img_w: embedding dim
        :param filter_windows: multi filter height
        :param filter_num: feature maps
        """
        self.inception_module_layers = []
        self.num_feature_maps = len(filter_windows) * filter_num
        for i, filter_h in enumerate(filter_windows):
            filter_w = img_w
            with tf.name_scope("conv-maxpool-%s" % filter_h):
                # Convolution Layer
                filter_shape = [filter_h, filter_w, 1, filter_num]
                pool_size = [1, img_h - filter_h + 1, img_w - filter_w + 1, 1]
                conv_layer = CNNModule(filter_num, filter_shape, pool_size)
                self.inception_module_layers.append(conv_layer)

    def __call__(self, q_input, a_input, *args, **kwargs):
        """
        concat outputs of multi-cnn-layer(conv+max_pool) with q_input vec and a_input vec.
        :param q_input: q_input vec
        :param a_input: a_input vec
        :return:
        """
        q_pooled_outputs = []
        a_pooled_outputs = []
        for conv_layer in self.inception_module_layers:
            q_pooled, a_pooled = conv_layer(q_input, a_input)
            q_pooled_outputs.append(q_pooled)
            a_pooled_outputs.append(a_pooled)
        # batch_size * num_filters
        q_sentence_vec = tf.reshape(tf.concat(q_pooled_outputs, 3), [-1, self.num_feature_maps])  # flatten and concat
        a_sentence_vec = tf.reshape(tf.concat(a_pooled_outputs, 3), [-1, self.num_feature_maps])
        return q_sentence_vec, a_sentence_vec


class QACNNModel(object):
    def __init__(self, word_embedding, img_h, img_w, filter_windows, feature_maps, n_in, n_hidden, n_out,
                 L1_reg, L2_reg, learning_rate, non_static=False):
        """
        question-answer cnn model init and definition.
        :param word_embedding: word embedding
        :param img_h: max sentence length.
        :param img_w: embedding dim.
        :param filter_windows: filter height, e.g [1,2,3]
        :param feature_maps: filter_num.
        :param n_in: mlp input size.
        :param n_hidden: mlp hidden size.
        :param n_out: mlp out size.
        :param L1_reg: mlp L1 loss.
        :param L2_reg: mlp L2 loss.
        :param learning_rate: learning rate for update.
        :param non_static: bool, update embedding or not.
        """
        self.lr = learning_rate
        self.word_embedding = word_embedding
        self.num_feature_maps = feature_maps * len(filter_windows)
        # define the placeholder
        with tf.name_scope('placeholder'):
            self.q_input = tf.placeholder(tf.int64, shape=[None, img_h], name='query_input')
            self.a_input = tf.placeholder(tf.int64, shape=[None, img_h], name='answer_input')
            self.l_input = tf.placeholder(tf.int64, shape=[None], name='label_input')  # one-hot -> [batch_size, n_out]
            self.keep_prop = tf.placeholder(tf.float32, name="keep_prop")  # drop
        # transfer input to vec with embedding.
        with tf.name_scope("embedding"):
            _word_embedding = tf.get_variable(name='word_emb', shape=self.word_embedding.shape, dtype=tf.float32,
                                              initializer=tf.constant_initializer(self.word_embedding), trainable=non_static)
            q_embedding = tf.nn.embedding_lookup(_word_embedding, self.q_input)
            a_embedding = tf.nn.embedding_lookup(_word_embedding, self.a_input)
            q_embedding_expanded = tf.expand_dims(q_embedding, -1)
            a_embedding_expanded = tf.expand_dims(a_embedding, -1)
            print "input shape(embedding expanded): ", q_embedding_expanded.get_shape()
        # define cnn model for qa.
        with tf.variable_scope("model_layers"):
            inception_module = InceptionModule(img_h, img_w, filter_windows, feature_maps)
            q_sentence_vec, a_sentence_vec = inception_module(q_embedding_expanded, a_embedding_expanded)
            interact_layer = InteractLayer(self.num_feature_maps, self.num_feature_maps, dim=n_in)
            qa_vec = interact_layer(q_sentence_vec, a_sentence_vec)
            bn_layer = BatchNormLayer(n_in=n_in, inputs=qa_vec)
        # define the classifier.
        with tf.name_scope("mlp"):
            classifier = MLP(bn_layer.out, n_in, n_hidden, n_out)
            # classifier = MLPDropout(bn_layer.out, n_in, n_hidden, n_out, keep_prop=self.keep_prop)
            # define cost, optimizer and output.
            self.pred_prob = classifier.pred_prob()
            self.error = classifier.errors(self.l_input)
            self.cost = classifier.cross_entropy(self.l_input) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
            self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.9).minimize(self.cost)

    def train_batch(self, sess, qsent_batch, asent_batch, label_batch, keep_prop):
        """
        qa_cnn model with mini_batch for train.
        :param sess: tensorflow session
        :param qsent_batch: batch of question.
        :param asent_batch: batch of answer.
        :param label_batch: batch of label
        :param keep_prop: keep prop for dropout.
        :return:
        """
        res = [self.optimizer, self.cost]
        _, cost = sess.run(res, feed_dict={self.q_input: qsent_batch,
                                           self.a_input: asent_batch,
                                           self.l_input: label_batch,
                                           self.keep_prop: keep_prop
                                           })
        return cost

    def eval_batch(self, sess, qsent_batch, asent_batch, label_batch, keep_prop):
        """
        qa_cnn model with mini_batch for eval(valid|test).
        :param sess: tensorflow session
        :param qsent_batch: batch of question.
        :param asent_batch: batch of answer.
        :param label_batch: batch of label
        :param keep_prop: keep prop for dropout.
        :return:
        """
        res = [self.pred_prob]
        pred = sess.run(res, feed_dict={self.q_input: qsent_batch,
                                        self.a_input: asent_batch,
                                        self.l_input: label_batch,
                                        self.keep_prop: keep_prop
                                        })[0]
        return pred

