#-*- coding:utf-8 -*-
__author__ = "ChenJun"

import numpy as np
import tensorflow as tf
import cPickle as pickle
from data_process.load_data import data_loader
from tensorflow_models.qa_cnn import QACNNModel
from qa_score import qa_evaluate


def get_overlap_feature(path, length):
    train_overlap = pickle.load(open(path + "pkl/overlap01-train.pkl", "r"))
    valid_overlap = pickle.load(open(path + "pkl/overlap01-valid.pkl", "r"))
    test_overlap = pickle.load(open(path + "pkl/overlap01-test.pkl", "r"))

    train_overlap_q, train_overlap_a = train_overlap[:, 0], train_overlap[:, 1]
    valid_overlap_q, valid_overlap_a = valid_overlap[:, 0], valid_overlap[:, 1]
    test_overlap_q, test_overlap_a = test_overlap[:, 0], test_overlap[:, 1]
    print "overlap01 feature shape: ", train_overlap_q.shape, valid_overlap_q.shape, test_overlap_a.shape

    train_overlap_q = train_overlap_q.reshape((train_overlap.shape[0], 1, length, 1))
    train_overlap_a = train_overlap_a.reshape((train_overlap.shape[0], 1, length, 1))
    valid_overlap_q = valid_overlap_q.reshape((valid_overlap.shape[0], 1, length, 1))
    valid_overlap_a = valid_overlap_a.reshape((valid_overlap.shape[0], 1, length, 1))
    test_overlap_q = test_overlap_q.reshape((test_overlap.shape[0], 1, length, 1))
    test_overlap_a = test_overlap_a.reshape((test_overlap.shape[0], 1, length, 1))

    return [(train_overlap_q, valid_overlap_q, test_overlap_q), (train_overlap_a, valid_overlap_a, test_overlap_a)]


def train_cnn(batch_size, n_epochs, img_h, img_w, filter_windows, feature_maps, n_in, n_hidden, n_out,
              L1_reg, L2_reg, learning_rate, random=False, non_static=False):
    ###############
    # LOAD DATA   #
    ###############
    print "loading the data... "
    path = "/Users/chenjun/PycharmProjects/DBQA/"
    loader = data_loader(path + "pkl/data-train-nn.pkl", path + "pkl/data-valid-nn.pkl", path + "pkl/data-test-nn.pkl", path + "/pkl/index2vec.pkl",)
    valid_group_list = pickle.load(open(path+"pkl/valid_group.pkl"))
    test_group_list = [int(x.strip()) for x in open(path+"data/dbqa-data-test.txt.group")]

    datasets, emb_words = loader.get_input_by_model(model="tensorflow",random=random)
    train_q_data, valid_q_data, test_q_data = datasets[0]
    train_a_data, valid_a_data, test_a_data = datasets[1]
    train_l_data, valid_l_data, test_l_data = datasets[2]

    # calculate the number of batches.
    n_train_batches = train_q_data.shape[0] // batch_size
    n_valid_batches = valid_q_data.shape[0] // batch_size
    n_test_batches = test_q_data.shape[0] // batch_size
    print "batch_size: %i, n_train_batches: %i, n_train_batches: %i, n_test_batches: %i" % (batch_size, n_train_batches, n_valid_batches, n_test_batches)
    ###############
    # BUILD MODEL #
    ###############
    print "building the model... "
    cnn_model = QACNNModel(word_embedding=emb_words, img_h=img_h, img_w=img_w, filter_windows=filter_windows,
                           feature_maps=feature_maps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, L1_reg=L1_reg, L2_reg=L2_reg,
                           learning_rate=learning_rate, non_static=non_static)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    ###############
    # TRAIN MODEL #
    ###############
    print "training the model... "
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        batch_cost = 0.
        for batch_index1 in xrange(n_train_batches):
            qsent_batch = train_q_data[batch_size * batch_index1: batch_size * (batch_index1 + 1)]
            asent_batch = train_a_data[batch_size * batch_index1: batch_size * (batch_index1 + 1)]
            label_batch = train_l_data[batch_size * batch_index1: batch_size * (batch_index1 + 1)]
            batch_cost += cnn_model.train_batch(sess=sess, qsent_batch=qsent_batch, asent_batch=asent_batch,
                                                label_batch=label_batch, keep_prop=0.5)
            # validation & test
            if batch_index1 % 100 == 0:
                print ("epoch %i/%i, batch %d/%d, cost %f") % (
                epoch, n_epochs, batch_index1, n_train_batches, batch_cost / n_train_batches)
                valid_score_data = []
                for batch_index in xrange(n_valid_batches):
                    qsent_batch = valid_q_data[batch_size * batch_index: batch_size * (batch_index + 1)]
                    asent_batch = valid_a_data[batch_size * batch_index: batch_size * (batch_index + 1)]
                    label_batch = valid_l_data[batch_size * batch_index: batch_size * (batch_index + 1)]
                    valid_pred = cnn_model.eval_batch(sess=sess, qsent_batch=qsent_batch, asent_batch=asent_batch,
                                                      label_batch=label_batch, keep_prop=1.0)
                    valid_score_data.append(valid_pred)
                valid_score_list = (np.concatenate(np.asarray(valid_score_data), axis=0)).tolist()
                valid_label_list = valid_l_data.tolist()
                for i in xrange(len(valid_score_list), len(valid_label_list)):
                    valid_score_list.append(np.random.random())
                _eval = qa_evaluate(valid_score_list, valid_label_list, valid_group_list, label=1, mod="mrr")  # one-hot -> label=[0,1]
                print "---valid mrr: ", _eval

                test_score_data = []
                for batch_index in xrange(n_test_batches):
                    qsent_batch = test_q_data[batch_size * batch_index: batch_size * (batch_index + 1)]
                    asent_batch = test_a_data[batch_size * batch_index: batch_size * (batch_index + 1)]
                    label_batch = test_l_data[batch_size * batch_index: batch_size * (batch_index + 1)]
                    test_pred = cnn_model.eval_batch(sess=sess, qsent_batch=qsent_batch, asent_batch=asent_batch,
                                                     label_batch=label_batch, keep_prop=1.0)
                    test_score_data.append(test_pred)
                test_score_list = (np.concatenate(np.asarray(test_score_data), axis=0)).tolist()
                test_label_list = test_l_data.tolist()
                for i in xrange(len(test_score_list), len(test_label_list)):
                    test_score_list.append(np.random.random())
                _eval = qa_evaluate(test_score_list, test_label_list, test_group_list, label=1, mod="mrr")  # one-hot -> label=[0,1]
                print "---error mrr: ", _eval


if __name__ == "__main__":
    train_cnn(batch_size=64,
              n_epochs=3,
              img_h=50,
              img_w=50,
              filter_windows=[1, 2, 3],
              feature_maps=128,
              n_in=20,
              n_hidden=128,
              n_out=2,
              L1_reg=0.00,
              L2_reg=0.0001,
              learning_rate=0.01,
              random=False,
              non_static=False)
