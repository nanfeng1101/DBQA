#-*- coding:utf-8 -*-
__author__ = "ChenJun"
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from theano_models.qa_cnn import CNNModule
from theano_models.layers import InteractLayer, MLP, MLPDropout, BatchNormLayer
from theano_models.optimizer import Optimizer
from data_process.load_data import data_loader
from collections import OrderedDict
from qa_score import qa_evaluate
from weighted_model import ensemble
import warnings
warnings.filterwarnings("ignore")

SEED = 3435
rng = np.random.RandomState(SEED)

def get_overlap(path,length):
    train_overlap = pickle.load(open(path+"pkl/overlap01-train.pkl","r"))
    valid_overlap = pickle.load(open(path+"pkl/overlap01-valid.pkl","r"))
    test_overlap = pickle.load(open(path+"pkl/overlap01-test.pkl","r"))

    train_overlap_q, train_overlap_a = train_overlap[:,0], train_overlap[:,1]
    valid_overlap_q, valid_overlap_a = valid_overlap[:,0], valid_overlap[:,1]
    test_overlap_q, test_overlap_a = test_overlap[:,0], test_overlap[:,1]
    print "overlap01 feature shape: ", train_overlap_q.shape, valid_overlap_q.shape, test_overlap_a.shape

    train_overlap_q = theano.shared(value=train_overlap_q.reshape((train_overlap.shape[0], 1, length,1)),borrow=True)
    train_overlap_a = theano.shared(value=train_overlap_a.reshape((train_overlap.shape[0], 1, length, 1)), borrow=True)
    valid_overlap_q = theano.shared(value=valid_overlap_q.reshape((valid_overlap.shape[0], 1, length, 1)), borrow=True)
    valid_overlap_a = theano.shared(value=valid_overlap_a.reshape((valid_overlap.shape[0], 1, length, 1)), borrow=True)
    test_overlap_q = theano.shared(value=test_overlap_q.reshape((test_overlap.shape[0], 1, length, 1)), borrow=True)
    test_overlap_a = theano.shared(value=test_overlap_a.reshape((test_overlap.shape[0], 1, length, 1)), borrow=True)

    return [(train_overlap_q,valid_overlap_q,test_overlap_q),(train_overlap_a,valid_overlap_a,test_overlap_a)]


def build_model(batch_size,img_h,img_w,filter_windows,filter_num,n_in,n_hidden,n_out,L1_reg,L2_reg,conv_non_linear,learning_rate,n_epochs,random=False,non_static=False):
    """
    build cnn model for QA.
    :param batch_size: batch_size
    :param img_h: sentence length
    :param img_w: word vector dimension [100] 
    :param filter_windows: filter window sizes 
    :param filter_num: the number of feature maps (per filter window)
    :param n_in: num of input units
    :param n_hidden: num of hidden units
    :param n_out: num of out units
    :param L1_reg: mlp L1 loss
    :param L2_reg: mlp L2 loss
    :param conv_non_linear: activation
    :param learning_rate: learning rate
    :param n_epochs: num of epochs
    :param random: bool, use random embedding or trained embedding
    :param non_static: bool, use word embedding for param or not
    :return: 
    """

    global rng
    ###############
    # LOAD DATA   #
    ###############
    print "loading the data... "
    path = "/Users/chenjun/PycharmProjects/DBQA/"
    loader = data_loader(path+"pkl/data-train-nn.pkl",path+"pkl/data-valid-nn.pkl",path+"pkl/data-test-nn.pkl", path+"pkl/index2vec.pkl")
    valid_group_list = pickle.load(open(path+"pkl/valid_group.pkl"))
    test_group_list = [int(x.strip()) for x in open(path + "data/dbqa-data-test.txt.group")]

    datasets, emb_words = loader.get_input_by_model(model="theano",random=random)
    train_q_data, valid_q_data, test_q_data = datasets[0]
    train_a_data, valid_a_data, test_a_data = datasets[1]
    train_l_data, valid_l_data, test_l_data = datasets[2]

    features = get_overlap(path, length=img_h)
    train_overlap_q, valid_overlap_q, test_overlap_q = features[0]
    train_overlap_a, valid_overlap_a, test_overlap_a = features[1]

    # calculate the number of batches
    n_train_batches = train_q_data.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_q_data.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_q_data.get_value(borrow=True).shape[0] // batch_size
    print "batch_size: %i, n_train_batches: %i, n_valid_batches: %i, n_test_batches: %i" % (batch_size, n_train_batches, n_valid_batches, n_test_batches)

    ###############
    # BUILD MODEL #
    ###############
    print "building the model... "
    # define the input variable
    index = T.lscalar(name="index")
    drop_rate = T.fscalar(name="drop_rate")
    x1 = T.matrix(name='x1', dtype='int64')
    x2 = T.matrix(name='x2', dtype='int64')
    y = T.lvector(name='y')
    x1_overlap = T.tensor4(name="x1_overlap", dtype='float32')
    x2_overlap = T.tensor4(name="x2_overlap", dtype='float32')

    # transfer input to vector with embedding.
    _x1 = emb_words[x1.flatten()].reshape((x1.shape[0], 1, img_h, img_w - 1))
    emb_x1 = T.concatenate([_x1, x1_overlap], axis=3)
    _x2 = emb_words[x2.flatten()].reshape((x2.shape[0], 1, img_h, img_w - 1))
    emb_x2 = T.concatenate([_x2, x2_overlap], axis=3)

    # conv_layer
    conv_layers = []
    q_input = []
    a_input = []
    for i, filter_h in enumerate(filter_windows):
        filter_w = img_w
        filter_shape = (filter_num, 1, filter_h, filter_w)
        pool_size = (img_h - filter_h + 1, img_w - filter_w + 1)
        conv_layer = CNNModule(rng, filter_shape=filter_shape, pool_size=pool_size, non_linear=conv_non_linear)
        q_conv_output, a_conv_output = conv_layer(emb_x1, emb_x2)
        q_conv_output = q_conv_output.flatten(2)  # [batch_size * filter_num]
        a_conv_output = a_conv_output.flatten(2)  # [batch_size * filter_num]
        q_input.append(q_conv_output)
        a_input.append(a_conv_output)
        conv_layers.append(conv_layer)
    q_input = T.concatenate(q_input, axis=1)  # batch_size*(filter_num*len(filter_windows))
    a_input = T.concatenate(a_input, axis=1)  # batch_size*(filter_num*len(filter_windows))
    num_filters = len(filter_windows) * filter_num
    interact_layer = InteractLayer(rng, num_filters, num_filters, dim=n_in)
    qa_vec = interact_layer(q_input, a_input)
    bn_layer = BatchNormLayer(n_in=n_in, inputs=qa_vec)

    # classifier = MLP(rng,input=bn_layer.out,n_in=n_in,n_hidden=n_hidden,n_out=n_out)
    classifier = MLPDropout(rng, input=bn_layer.out, n_in=n_in, n_hidden=n_hidden, n_out=n_out, dropout_rate=drop_rate)
    # model params
    params = classifier.params + interact_layer.params + bn_layer.params
    for i in xrange(len(conv_layers)):
        params += conv_layers[i].params
    if non_static:
        print "---CNN-NON-STATIC---"
        params += [emb_words]
    else:
        print "---CNN-STATIC---"

    opt = Optimizer()
    cost = (
        classifier.cross_entropy(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # updates = opt.sgd_updates_adadelta(params, cost, 0.95, 1e-6, 9)
    updates = opt.RMSprop(params, cost)

    train_model = theano.function(
        inputs=[index, drop_rate],
        updates=updates,
        outputs=cost,
        givens={
            x1: train_q_data[index * batch_size:(index + 1) * batch_size],
            x2: train_a_data[index * batch_size:(index + 1) * batch_size],
            y: train_l_data[index * batch_size:(index + 1) * batch_size],
            x1_overlap: train_overlap_q[index * batch_size: (index + 1) * batch_size],
            x2_overlap: train_overlap_a[index * batch_size: (index + 1) * batch_size]
        },
    )
    valid_model = theano.function(
        inputs=[index, drop_rate],
        outputs=classifier.pred_prob(),
        givens={
            x1: valid_q_data[index * batch_size:(index + 1) * batch_size],
            x2: valid_a_data[index * batch_size:(index + 1) * batch_size],
            x1_overlap: valid_overlap_q[index * batch_size: (index + 1) * batch_size],
            x2_overlap: valid_overlap_a[index * batch_size: (index + 1) * batch_size]
        },
    )
    test_model = theano.function(
        inputs=[index, drop_rate],
        outputs=classifier.pred_prob(),
        givens={
            x1: test_q_data[index * batch_size:(index + 1) * batch_size],
            x2: test_a_data[index * batch_size:(index + 1) * batch_size],
            x1_overlap: test_overlap_q[index * batch_size: (index + 1) * batch_size],
            x2_overlap: test_overlap_a[index * batch_size: (index + 1) * batch_size]
        },
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
                # VALID MODEL #
                ###############
                valid_score_data = []
                for batch_index2 in xrange(n_valid_batches):
                    batch_pred = valid_model(batch_index2, 0.0)  # drop
                    valid_score_data.append(batch_pred)
                valid_score_list = (np.concatenate(np.asarray(valid_score_data), axis=0)).tolist()
                valid_label_list = valid_l_data.get_value(borrow=True).tolist()
                for i in xrange(len(valid_score_list), len(valid_label_list)):
                    valid_score_list.append(np.random.random())
                _eval = qa_evaluate(valid_score_list, valid_label_list, valid_group_list, label=1, mod="mrr")
                print "---valid mrr: ", _eval
                valid_dic[str(epoch) + "-" + str(batch_index1)] = _eval
                ###############
                # TEST MODEL  #
                ###############
                test_score_data = []
                for batch_index3 in xrange(n_test_batches):
                    batch_pred = test_model(batch_index3, 0.0)  # drop
                    test_score_data.append(batch_pred)
                test_score_list = (np.concatenate(np.asarray(test_score_data), axis=0)).tolist()
                test_label_list = test_l_data.get_value(borrow=True).tolist()
                for i in xrange(len(test_score_list), len(test_label_list)):
                    test_score_list.append(np.random.random())
                _eval = qa_evaluate(test_score_list, test_label_list, test_group_list, label=1, mod="mrr")
                print "---test mrr: ", _eval
                eval_dic[str(epoch) + "-" + str(batch_index1)] = _eval
                pickle.dump(valid_score_list, open(path + "result/cnn-overlap-valid.pkl." + str(epoch) + "-" + str(batch_index1), "w"))
                pickle.dump(test_score_list, open(path + "result/cnn-overlap-test.pkl."+str(epoch)+"-"+str(batch_index1), "w"))
                pickle.dump(test_label_list, open(path + "result/test_label.pkl", "w"))
                pickle.dump(valid_label_list, open(path + "result/valid_label.pkl", "w"))
        _valid_dic = sorted(valid_dic.items(), key=lambda x: x[1])[-10:]
        _eval_dic = sorted(eval_dic.items(), key=lambda x: x[1])[-10:]
        print "valid dic: ", _valid_dic
        print "eval dic: ", _eval_dic
        valid_score_file = [path+"result/cnn-overlap-valid.pkl."+x[0] for x in _valid_dic]
        test_score_file = [path + "result/cnn-overlap-test.pkl." + x[0] for x in _valid_dic] ###from valid
        valid_label_file = path + "result/valid_label.pkl"
        test_label_file = path + "result/test_label.pkl"
        test_ensemble_file = path + "result/test_ensemble_overlap.pkl"
        valid_ensemble_file = path + "result/valid_ensemble_overlap.pkl"
        valid_mrr = ensemble(valid_score_file, valid_label_file, valid_group_list, valid_ensemble_file)
        test_mrr = ensemble(test_score_file, test_label_file, test_group_list, test_ensemble_file)
        print "---ensemble valid mrr: ", valid_mrr
        print "---ensemble test mrr: ", test_mrr

if __name__ == "__main__":
    build_model(batch_size=64,
                img_h=50,
                img_w=51,
                filter_windows=[3, 1, 2],
                filter_num=128,
                n_in=20,
                n_hidden=128,
                n_out=2,
                L1_reg=0.00,
                L2_reg=0.0001,
                conv_non_linear="relu",
                learning_rate=0.001,
                n_epochs=3,
                random=False,
                non_static=False)

