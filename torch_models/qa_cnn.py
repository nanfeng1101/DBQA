# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import torch
import torch.nn as nn
from torch_models.layers import InteractLayer, BatchNormLayer, MLP, MLPDropout


class CNNModule(nn.Module):
    """
    qa_cnn module.
    """
    def __init__(self, feature_maps, filter_shape, pool_size, channels=1):
        """
        qa_cnn module init.
        :param feature_maps: feature maps(filter_num) after convolution.
        :param filter_shape: filter shape for convolution.
        :param pool_size: pool size for max pooling.
        :param channels: in channels, default=1.
        """
        super(CNNModule, self).__init__()
        self.cnn_layer = nn.Sequential(nn.Conv2d(channels, feature_maps, filter_shape),
                                       nn.ReLU(),
                                       nn.MaxPool2d(pool_size)
                                       )

    def forward(self, q_input, a_input):
        """
        convolution + max_pool for q_input and a_input.
        :param q_input: q_input vec.
        :param a_input: a_inut vec.
        :return:
        """
        q_out = self.cnn_layer(q_input)
        a_out = self.cnn_layer(a_input)
        return q_out, a_out


class InceptionModule(nn.Module):
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
        super(InceptionModule, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.filter_windows = filter_windows
        self.filter_num = filter_num
        self.num_feature_maps = len(filter_windows) * filter_num
        self.layers_num, self.filter_shapes, self.pool_sizes = self.param()
        for i, filter_shape, pool_size in zip(self.layers_num, self.filter_shapes, self.pool_sizes):
            self.add_module(name="cnn_layer_{}".format(i), module=CNNModule(self.filter_num, filter_shape, pool_size))

    def param(self):
        """
        get param(filter_shape and pool_size) for cnn module.
        :return:
        """
        filter_shapes = []
        pool_sizes = []
        layers_num = []
        for i, filter_h in enumerate(self.filter_windows):
            filter_shapes.append((filter_h, self.img_w))
            pool_sizes.append((self.img_h - filter_h + 1, 1))
            layers_num.append(i)
        return layers_num, filter_shapes, pool_sizes

    def forward(self, q_input, a_input):
        """
        concat outputs of multi-cnn-layer(conv+max_pool) with q_input vec and a_input vec.
        :param q_input: q_input vec
        :param a_input: a_input vec
        :return:
        """
        q_output, a_output = [], []
        for cnn_layer in self.children():
            q_out, a_out = cnn_layer(q_input, a_input)
            q_output.append(q_out)
            a_output.append(a_out)
        q_vec = torch.cat(q_output, dim=1).view(-1, self.num_feature_maps)  # batch * num_feature_maps
        a_vec = torch.cat(a_output, dim=1).view(-1, self.num_feature_maps)
        return q_vec, a_vec


class QACNNModel(nn.Module):
    """
    cnn model for QA pair.
    """
    def __init__(self, embedding, img_h, img_w, filter_windows, filter_num, n_in, n_hidden, n_out):
        """
        model init.
        :param embedding: word embedding.
        :param img_h: sentence length.
        :param img_w: embedding dim.
        :param filter_windows: collection of filter height.
        :param filter_num: feature maps.
        :param n_in: input size for mlp
        :param n_hidden: hidden size for mlp
        :param n_out: out size for mlp
        """
        super(QACNNModel, self).__init__()
        self.embedding = embedding
        self.img_h = img_h
        self.img_w = img_w
        self.filter_windows = filter_windows
        self.filter_num = filter_num
        self.input_size = n_in
        self.hidden_size = n_hidden
        self.out_size = n_out
        self.num_feature_maps = len(self.filter_windows) * self.filter_num
        self.inception_module_layers = InceptionModule(self.img_h, self.img_w, self.filter_windows, self.filter_num)
        self.interact_layer = InteractLayer(self.num_feature_maps, self.num_feature_maps, self.input_size)
        self.bn_layer = BatchNormLayer(self.input_size)
        self.mlp = MLPDropout(self.input_size, self.hidden_size, self.out_size)

    def forward(self, q_input, a_input, drop_rate):
        """
        input -> embedding_layer -> multi_cnn_layer -> interact_layer -> batchnorm_layer -> mlp_layer
        :param q_input: question sentence vec
        :param a_input: answer sentence vec
        :param: drop_rate: dropout rate
        :return:
        """
        q_input_emb = torch.unsqueeze(self.embedding(q_input), dim=1)
        a_input_emb = torch.unsqueeze(self.embedding(a_input), dim=1)
        q_vec, a_vec = self.inception_module_layers(q_input_emb, a_input_emb)
        qa_vec = self.interact_layer(q_vec, a_vec)
        bn_vec = self.bn_layer(qa_vec)
        prop, cate = self.mlp(bn_vec, drop_rate)
        return prop, cate
