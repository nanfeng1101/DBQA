# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BatchNormLayer(nn.Module):
    """
    batch normalization.
    """
    def __init__(self, n_in):
        super(BatchNormLayer, self).__init__()
        self.bn = nn.BatchNorm1d(n_in)

    def forward(self, inputs):
        out = self.bn(inputs)
        return out


class InteractLayer(nn.Module):
    """
    interact layer for q_sentence_vec and a_sentence_vec.
    math: (q * W) * a
    q * W: [batch_size, n_q] * [n_q, dim, n_a] -> [batch_size, dim, n_a]
    (q * W) * a: [batch_size, dim, n_a] * [batch_size, n_a, 1] -> [batch_size, dim, 1]
    out: [batch_size, dim, 1] -> [batch_size, dim]
    """
    def __init__(self, n_q, n_a, dim):
        super(InteractLayer, self).__init__()
        self.W = Variable(torch.randn(n_q, dim, n_a) * 0.05)
        self.dim = dim
        self.input_size = n_q

    def forward(self, q_input, a_input):
        qw = torch.mm(q_input, self.W.view(self.input_size, -1)).view(-1, self.dim, self.input_size)
        qwa = torch.bmm(qw, torch.unsqueeze(a_input, 2))
        qa_vec = qwa.view(-1, self.dim)
        return qa_vec


class MLPDropout(nn.Module):
    """
    mlp layer for predict and classify.
    hidden layer + dropout + logistic layer
    """
    def __init__(self, n_in, n_hidden, n_out):
        super(MLPDropout, self).__init__()
        self.hidden_layer = nn.Linear(n_in, n_hidden)
        self.tanh = nn.Tanh()
        self.logistic_layer = nn.Linear(n_hidden, n_out)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs, drop_rate):
        hidden_out = self.tanh(self.hidden_layer(inputs))
        drop_out = F.dropout(hidden_out, p=drop_rate)
        pred_prop = self.softmax(self.logistic_layer(drop_out))
        _, pred_cate = torch.max(pred_prop, dim=1)
        return pred_prop, pred_cate


class MLP(nn.Module):
    """
    mlp layer for predict and classify.
    hidden layer + logistic layer
    """
    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, n_hidden),
                                 nn.Tanh(),
                                 nn.Linear(n_hidden, n_out),
                                 nn.LogSoftmax()
                                 )

    def forward(self, inputs):
        pred_prop = self.mlp(inputs)
        _, pred_cate = torch.max(pred_prop, dim=1)
        return pred_prop, pred_cate