# -*- coding: utf-8 -*-
__author__ = 'Chenjun'
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
from collections import OrderedDict
from data_process.load_data import data_loader
from theano_models.layers import MLP, InteractLayer, BatchNormLayer, MLPDropout
from theano_models.qa_rnn import LSTM, GRU
from theano_models.optimizer import Optimizer
from qa_score import qa_evaluate
import warnings
warnings.filterwarnings("ignore")

SEED = 1234
rng = np.random.RandomState(SEED)


def get_overlap(path, length):
    train_overlap = pickle.load(open(path+"pkl/overlap01-train.pkl","r"))
    valid_overlap = pickle.load(open(path+"pkl/overlap01-valid.pkl","r"))
    test_overlap = pickle.load(open(path+"pkl/overlap01-test.pkl","r"))

    train_overlap_q, train_overlap_a = train_overlap[:,0], train_overlap[:,1]
    valid_overlap_q, valid_overlap_a = valid_overlap[:,0], valid_overlap[:,1]
    test_overlap_q, test_overlap_a = test_overlap[:,0], test_overlap[:,1]
    print "overlap01 feature shape: ", train_overlap_q.shape, valid_overlap_q.shape, test_overlap_a.shape

    train_overlap_q = theano.shared(value=train_overlap_q.reshape((train_overlap.shape[0], length, 1)),borrow=True)
    train_overlap_a = theano.shared(value=train_overlap_a.reshape((train_overlap.shape[0], length, 1)), borrow=True)
    valid_overlap_q = theano.shared(value=valid_overlap_q.reshape((valid_overlap.shape[0], length, 1)), borrow=True)
    valid_overlap_a = theano.shared(value=valid_overlap_a.reshape((valid_overlap.shape[0], length, 1)), borrow=True)
    test_overlap_q = theano.shared(value=test_overlap_q.reshape((test_overlap.shape[0], length, 1)), borrow=True)
    test_overlap_a = theano.shared(value=test_overlap_a.reshape((test_overlap.shape[0], length, 1)), borrow=True)
    return [(train_overlap_q,valid_overlap_q,test_overlap_q),(train_overlap_a,valid_overlap_a,test_overlap_a)]


def build_model(batch_size, length, emb_dim, n_in, n_hidden_lstm, n_in_mlp, n_hidden_mlp, n_out, L1_reg, L2_reg,
                learning_rate, n_epochs, random, non_static):
    """
    rnn model for QA pair.
    :param batch_size: batch size
    :param length: sentence length
    :param emb_dim: embedding dim
    :param n_in: rnn layer input units
    :param n_hidden_lstm: rnn layer hidden units
    :param n_in_mlp: mlp input units
    :param n_hidden_mlp: mlp hidden units
    :param n_out: mlp out size
    :param L1_reg: mlp L1 loss
    :param L2_reg: mlp L2 loss
    :param learning_rate: learning rate for update
    :param n_epochs: epoch num
    :param random: bool, use trained embedding or random embedding
    :param non_static: bool, update embedding or not
    :return:
    """
    global rng
    ###############
    # LOAD DATA   #
    ###############
    print "loading the data... "
    path = "/Users/chenjun/PycharmProjects/DBQA/"
    loader = data_loader(path + "pkl/data-train-nn.pkl", path + "pkl/data-valid-nn.pkl", path + "pkl/data-test-nn.pkl",
                         path + "pkl/index2vec.pkl")
    valid_group_list = pickle.load(open(path + "pkl/valid_group.pkl"))
    test_group_list = [int(x.strip()) for x in open(path + "data/dbqa-data-test.txt.group")]

    datasets, emb_words = loader.get_input_by_model(model="theano", random=random)
    train_q_data, valid_q_data, test_q_data = datasets[0]
    train_a_data, valid_a_data, test_a_data = datasets[1]
    train_l_data, valid_l_data, test_l_data = datasets[2]

    features = get_overlap(path, length=length)
    train_overlap_q, valid_overlap_q, test_overlap_q = features[0]
    train_overlap_a, valid_overlap_a, test_overlap_a = features[1]

    # calculate the number of batches.
    n_train_batches = train_q_data.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_q_data.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_q_data.get_value(borrow=True).shape[0] // batch_size
    print "batch_size: %i, n_train_batches: %i, n_valid_batches: %i, n_test_batches: %i" % (batch_size, n_train_batches,n_valid_batches,n_test_batches)
    ###############
    # BUILD MODEL #
    ###############
    print "building the model... "
    # define input variable
    index = T.lscalar('index')
    drop_rate = T.fscalar("drop_rate")
    x1 = T.matrix('x1', dtype='int64')  # batch_size*length
    x2 = T.matrix('x2', dtype='int64')  # batch_size*length
    y = T.lvector('y')  # batch_size*1
    x1_overlap = T.tensor3(name="x1_overlap", dtype='float32')
    x2_overlap = T.tensor3(name="x2_overlap", dtype='float32')

    _x1 = emb_words[x1.flatten()].reshape((x1.shape[0], length, emb_dim))
    emb_x1 = T.concatenate([_x1, x1_overlap], axis=2)
    _x2 = emb_words[x2.flatten()].reshape((x1.shape[0], length, emb_dim))
    emb_x2 = T.concatenate([_x2, x2_overlap], axis=2)

    # rnn layer
    lstm_layer = GRU(rng=rng, n_in=n_in, n_hidden=n_hidden_lstm)
    q_sentence_vec = lstm_layer(input=emb_x1)
    a_sentence_vec = lstm_layer(input=emb_x2)
    interact_layer = InteractLayer(rng, n_hidden_lstm, n_hidden_lstm, n_in_mlp)
    qa_vec = interact_layer(q_sentence_vec, a_sentence_vec)
    bn_layer = BatchNormLayer(n_in=n_in_mlp, inputs=qa_vec)
    # mlp classifier
    # mlp_layer = MLP(rng=rng, input=bn_layer.out, n_in=n_in_mlp, n_hidden=n_hidden_mlp, n_out=n_out)
    mlp_layer = MLPDropout(rng=rng, input=bn_layer.out, n_in=n_in_mlp, n_hidden=n_hidden_mlp, n_out=n_out, dropout_rate=drop_rate)
    if non_static:
        print "---RNN-NON-STATIC---"
        params = lstm_layer.params + interact_layer.params + mlp_layer.params + [emb_words]
    else:
        print "---RNN-STATIC---"
        params = lstm_layer.params + interact_layer.params + mlp_layer.params

    cost = (
        mlp_layer.cross_entropy(y)
        + L1_reg * mlp_layer.L1
        + L2_reg * mlp_layer.L2_sqr
    )
    opt = Optimizer()
    # updates = opt.sgd(params,cost,learning_rate)
    updates = opt.sgd_updates_adadelta(params, cost, 0.95, 1e-6, 9)

    train_model = theano.function(
        inputs=[index, drop_rate],
        outputs=cost,
        updates=updates,
        givens={
            x1: train_q_data[index * batch_size:(index + 1) * batch_size],
            x2: train_a_data[index * batch_size:(index + 1) * batch_size],
            y: train_l_data[index * batch_size:(index + 1) * batch_size],
            x1_overlap: train_overlap_q[index * batch_size: (index + 1) * batch_size],
            x2_overlap: train_overlap_a[index * batch_size: (index + 1) * batch_size]
        }
    )
    valid_model = theano.function(
        inputs=[index, drop_rate],
        outputs=mlp_layer.pred_prob(),
        givens={
            x1: valid_q_data[index * batch_size:(index + 1) * batch_size],
            x2: valid_a_data[index * batch_size:(index + 1) * batch_size],
            x1_overlap: valid_overlap_q[index * batch_size: (index + 1) * batch_size],
            x2_overlap: valid_overlap_a[index * batch_size: (index + 1) * batch_size]
        }
    )
    test_model = theano.function(
        inputs=[index, drop_rate],
        outputs=mlp_layer.pred_prob(),
        givens={
            x1: test_q_data[index * batch_size:(index + 1) * batch_size],
            x2: test_a_data[index * batch_size:(index + 1) * batch_size],
            x1_overlap: test_overlap_q[index * batch_size: (index + 1) * batch_size],
            x2_overlap: test_overlap_a[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('training the model...')
    epoch = 0
    valid_dic = OrderedDict()
    eval_dic = OrderedDict()
    while epoch < n_epochs:
        epoch += 1
        batch_cost = 0.
        for batch_index1 in xrange(n_train_batches):
            batch_cost += train_model(batch_index1, 0.5)  # drop
            if batch_index1 % 100 == 0:
                print ('epoch %i/%i, batch %i/%i, cost %f') % (
                    epoch, n_epochs, batch_index1, n_train_batches, batch_cost / n_train_batches)
                ###############
                # VALID MODEL  #
                ###############
                valid_score_data = []
                for batch_index in xrange(n_valid_batches):
                    batch_pred = valid_model(batch_index, 0.0)  # drop
                    valid_score_data.append(batch_pred)
                valid_score_list = (np.concatenate(np.asarray(valid_score_data), axis=0)).tolist()
                valid_label_list = valid_l_data.get_value(borrow=True).tolist()
                for i in xrange(len(valid_score_list), len(valid_label_list)):
                    valid_score_list.append(np.random.random())
                _eval = qa_evaluate(valid_score_list, valid_label_list, valid_group_list, label=1, mod="mrr")
                print "--valid mrr: ", _eval
                valid_dic[str(epoch)+"-"+str(batch_index1)] = _eval
                ###############
                # TEST MODEL  #
                ###############
                test_score_data = []
                for batch_index in xrange(n_test_batches):
                    batch_pred = test_model(batch_index, 0.0)  # drop
                    test_score_data.append(batch_pred)
                test_score_list = (np.concatenate(np.asarray(test_score_data), axis=0)).tolist()
                test_label_list = test_l_data.get_value(borrow=True).tolist()
                for i in xrange(len(test_score_list), len(test_label_list)):
                    test_score_list.append(np.random.random())
                _eval = qa_evaluate(test_score_list, test_label_list, test_group_list, label=1, mod="mrr")
                print "--test mrr: ", _eval
                eval_dic[str(epoch)+"-"+str(batch_index1)] = _eval
    print "valid dic: ", valid_dic
    print "eval dic: ", eval_dic

if __name__ == "__main__":
    build_model(batch_size=64,
                length=50,
                emb_dim=50,
                n_in=51,
                n_hidden_lstm=128,
                n_in_mlp=20,
                n_hidden_mlp=128,
                n_out=2,
                L1_reg=0.00,
                L2_reg=0.0001,
                learning_rate=0.001,
                n_epochs=3,
                random=False,
                non_static=False)

