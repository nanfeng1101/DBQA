# -*- coding: utf-8 -*-
__author__ = 'Chenjun'
import cPickle as pickle
import theano
import torch
import numpy as np


class data_loader(object):
    def __init__(self, train_data_file, valid_data_file, test_data_file, index2vec_file):
        self.train_data = pickle.load(open(train_data_file, 'r'))
        self.valid_data = pickle.load(open(valid_data_file, 'r'))
        self.test_data = pickle.load(open(test_data_file, 'r'))
        self.index2vec = pickle.load(open(index2vec_file, 'r'))

    def __split_data(self, data):
        """
        split data into query-data,answer-data,label-data,feature_data.
        :return: data int64
        """
        q_data, a_data, l_data = [], [], []
        for d in data:
            q_data.append(d[0])
            a_data.append(d[1])
            l_data.append(d[2])
        return q_data, a_data, l_data

    def __shared_data(self, data, borrow=True):
        """
        share data in theano.
        :param data: data
        :param borrow: True or false
        :return:
        """
        shared_data = theano.shared(value=data, borrow=borrow)
        return shared_data

    def get_input_by_model(self, model, random=False):
        """
        get query/answer/label data for train/validation/test
        :param model: theano | tensorflow | pytorch
        :param random: trained embedding | random embedding
        :return: 
        """
        train_q_data, train_a_data, train_l_data = self.__split_data(self.train_data)
        valid_q_data, valid_a_data, valid_l_data = self.__split_data(self.valid_data)
        test_q_data, test_a_data, test_l_data = self.__split_data(self.test_data)
        word_embedding = None
        print "---train data length: ", len(train_q_data)
        print "---valid data length: ", len(valid_q_data)
        print "---test  data length: ", len(test_q_data)
        if model == "theano":
            train_q_data = self.__shared_data(np.asarray(train_q_data, dtype='int64'), borrow=True)
            train_a_data = self.__shared_data(np.asarray(train_a_data, dtype='int64'), borrow=True)
            train_l_data = self.__shared_data(np.asarray(train_l_data, dtype='int64'), borrow=True)
            valid_q_data = self.__shared_data(np.asarray(valid_q_data, dtype='int64'), borrow=True)
            valid_a_data = self.__shared_data(np.asarray(valid_a_data, dtype='int64'), borrow=True)
            valid_l_data = self.__shared_data(np.asarray(valid_l_data, dtype='int64'), borrow=True)
            test_q_data = self.__shared_data(np.asarray(test_q_data, dtype='int64'), borrow=True)
            test_a_data = self.__shared_data(np.asarray(test_a_data, dtype='int64'), borrow=True)
            test_l_data = self.__shared_data(np.asarray(test_l_data, dtype='int64'), borrow=True)
            word_embedding = theano.shared(value=self.__get_word_embedding(random=random),name="emb_words",borrow=True)
        elif model == "tensorflow":
            # label = np.eye(2, dtype='int64')  # label[i] for one-hot
            train_l_data = np.asarray(train_l_data, dtype='int64')
            valid_l_data = np.asarray(valid_l_data, dtype='int64')
            test_l_data = np.asarray(test_l_data, dtype='int64')
            train_q_data = np.asarray(train_q_data, dtype='int64')
            valid_q_data = np.asarray(valid_q_data, dtype='int64')
            test_q_data = np.asarray(test_q_data, dtype='int64')
            train_a_data = np.asarray(train_a_data, dtype='int64')
            valid_a_data = np.asarray(valid_a_data, dtype='int64')
            test_a_data = np.asarray(test_a_data, dtype='int64')
            word_embedding = self.__get_word_embedding(random=random)
        elif model == "pytorch":
            train_l_data = torch.from_numpy(np.asarray(train_l_data, dtype='int64'))
            valid_l_data = torch.from_numpy(np.asarray(valid_l_data, dtype='int64'))
            test_l_data = torch.from_numpy(np.asarray(test_l_data, dtype='int64'))
            train_q_data = torch.from_numpy(np.asarray(train_q_data, dtype='int64'))
            valid_q_data = torch.from_numpy(np.asarray(valid_q_data, dtype='int64'))
            test_q_data = torch.from_numpy(np.asarray(test_q_data, dtype='int64'))
            train_a_data = torch.from_numpy(np.asarray(train_a_data, dtype='int64'))
            valid_a_data = torch.from_numpy(np.asarray(valid_a_data, dtype='int64'))
            test_a_data = torch.from_numpy(np.asarray(test_a_data, dtype='int64'))
            word_embedding = torch.from_numpy(self.__get_word_embedding(random=random))
        else:
            print ("model %s not supported...") % model
        dataset = [(train_q_data, valid_q_data, test_q_data), (train_a_data, valid_a_data, test_a_data), (train_l_data, valid_l_data, test_l_data)]
        return dataset, word_embedding

    def __get_data_mask(self, data):
        """
        get mask.
        :param data: data
        :return: data float32
        """
        data_mask = []
        for d in data:
            mask = [0.] * len(d)
            for i in xrange(len(d)):
                if d[i] != 0:
                    mask[i] = 1.
            data_mask.append(mask)
        return np.asarray(data_mask, dtype='float32')

    def __get_word_embedding(self, random):
        """
        get word_embedding.
        :return: data float32
        """
        word_embedding = np.asarray(self.index2vec, dtype='float32')  # trained
        if random:
            print "---use random embedding---"
            word_embedding = np.random.uniform(low=-0.1, high=0.1, size=word_embedding.shape).astype(dtype='float32')  # random
        else:
            print "---use trained embedding---"
        return word_embedding




