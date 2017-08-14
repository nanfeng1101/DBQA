# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_models.layers import InteractLayer, BatchNormLayer, MLP, MLPDropout


class LSTMModule(nn.Module):
    """
    qa lstm module.
    """
    def __init__(self, batch_size, n_in, n_hidden, num_layers=1):
        """
        lstm module init.
        :param batch_size: mini_batch size
        :param n_in: input size for lstm
        :param n_hidden: hidden size for rnn
        :param num_layers: num of layers
        """
        super(LSTMModule, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = n_in
        self.hidden_size = n_hidden
        self.lstm = nn.LSTM(input_size=n_in, hidden_size=n_hidden, num_layers=num_layers, batch_first=True)

    def init_hidden(self):
        """
        state(hidden state, cell state) init.
        :return:
        """
        h0 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        c0 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return (h0, c0)

    def forward(self, q_input, a_input):
        """
        lstm layer for q_input vec and a_input vec.
        :param q_input: q_input vec, [batch_size, sequence_length, emb_dim]
        :param a_input: a_input vec, [batch_size, sequence_length, emb_dim]
        :return:
        """
        q_out, _ = self.lstm(q_input, self.init_hidden())  # [batch_size, sequence_length, emb_dim]
        a_out, _ = self.lstm(a_input, self.init_hidden())
        return q_out[:, -1, :], a_out[:, -1, :]


class RNNModule(nn.Module):
    """
    qa rnn module.
    """
    def __init__(self, batch_size, n_in, n_hidden, num_layers=1):
        """
        rnn module init.
        :param batch_size: mini_batch size
        :param n_in: rnn input size
        :param n_hidden: rnn hidden size
        :param num_layers: num of layers
        """
        super(RNNModule, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = n_in
        self.hidden_size = n_hidden
        self.rnn = nn.RNN(input_size=n_in, hidden_size=n_hidden, num_layers=num_layers, batch_first=True)

    def init_hidden(self):
        """
        hidden state init.
        :return:
        """
        h0 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return h0

    def forward(self, q_input, a_input):
        """
        rnn later for input vec.
        :param q_input: q_input vec.
        :param a_input: a_input vec.
        :return:
        """
        q_out, _ = self.rnn(q_input, self.init_hidden())
        a_out, _ = self.rnn(a_input, self.init_hidden())
        return q_out[:, -1, :], a_out[:, -1, :]


class GRUModule(nn.Module):
    """
    qa gru module.
    """
    def __init__(self, batch_size, n_in, n_hidden, num_layers=1):
        """
        gru module init.
        :param batch_size: mini_batch size
        :param n_in: gru input size
        :param n_hidden: gru hidden size
        :param num_layers: num of layers
        """
        super(GRUModule, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = n_in
        self.hidden_size = n_hidden
        self.gru = nn.GRU(input_size=n_in, hidden_size=n_hidden, num_layers=num_layers, batch_first=True)

    def init_hidden(self):
        """
        hidden state init.
        :return:
        """
        h0 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return h0

    def forward(self, q_input, a_input):
        """
        gru layer for input vec, [batch_size, sequence_length, emb_dim].
        :param q_input: q_input vec
        :param a_input: a_input vec
        :return:
        """
        q_out, _ = self.gru(q_input, self.init_hidden())
        a_out, _ = self.gru(a_input, self.init_hidden())
        return q_out[:, -1, :], a_out[:, -1, :]


class QARNNModel(nn.Module):
    """
    RNN model for QA pair.
    """
    def __init__(self, embedding, batch_size, sequence_length, n_in_rnn, n_hidden_rnn, n_in_mlp, n_hidden_mlp, n_out):
        """
        RNN model init.
        :param embedding: word embedding
        :param batch_size: mini_batch size
        :param sequence_length: sequence_length
        :param n_in_rnn: input_size for rnn(emb_dim)
        :param n_hidden_rnn: hidden_size for rnn
        :param n_in_mlp: input_size for mlp
        :param n_hidden_mlp: hidden_size for mlp
        :param n_out: out_size for mlp
        """
        super(QARNNModel, self).__init__()
        self.embedding = embedding
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rnn_input_size = n_in_rnn
        self.rnn_hidden_size = n_hidden_rnn
        self.mlp_input_size = n_in_mlp
        self.mlp_hidden_size = n_hidden_mlp
        self.mlp_out_size = n_out
        self.rnn_layer = GRUModule(self.batch_size, self.rnn_input_size, self.rnn_hidden_size)
        self.interact_layer = InteractLayer(self.rnn_hidden_size, self.rnn_hidden_size, self.mlp_input_size)
        self.bn_layer = BatchNormLayer(self.mlp_input_size)
        self.mlp = MLPDropout(self.mlp_input_size, self.mlp_hidden_size, self.mlp_out_size)

    def forward(self, q_input, a_input, drop_rate):
        """
        embedding layer -> rnn layer -> interact layer -> batchnorm layer -> mlp layer
        :param q_input: question batch
        :param a_input: answer batch
        :param drop_rate: dropout rate
        :return:
        """
        q_input_emb = self.embedding(q_input)
        a_input_emb = self.embedding(a_input)
        q_vev, a_vec = self.rnn_layer(q_input_emb, a_input_emb)
        qa_vec = self.interact_layer(q_vev, a_vec)
        bn_vec = self.bn_layer(qa_vec)
        prop, cate = self.mlp(bn_vec, drop_rate)
        return prop, cate








