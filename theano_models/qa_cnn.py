#-*- coding:utf-8 -*-
__author__ = "ChenJun"
import numpy
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from layers import ReLU, Iden, Tanh, Sigmoid


class CNNModule(object):
    """cnn module for QA pair """
    def __init__(self, rng, filter_shape, pool_size, non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type q_input: theano.tensor.dtensor4
        :param q_input: symbolic variable that describes the question input of the
        architecture (one minibatch)
        
        :type a_input: theano.tensor.dtensor4
        :param a_input: symbolic variable that describes the answer input of the
        architecture (one minibatch)

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type pool_size: tuple or list of length 2
        :param pool_size: the downsampling (pooling) factor (#rows, #cols)
        """
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(pool_size))
        # initialize weights with random weights
        if non_linear == "none" or non_linear == "relu":
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True, name="W_conv")
            

        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype=theano.config.floatX),borrow=True, name="W")
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b")
        # store parameters of this layer
        self.params = [self.W, self.b]

    def __call__(self, q_input, a_input, *args, **kwargs):
        # convolve input feature maps with filters
        q_conv_out = conv2d(
            input=q_input,
            filters=self.W,
            filter_shape=self.filter_shape
        )
        a_conv_out = conv2d(
            input=a_input,
            filters=self.W,
            filter_shape=self.filter_shape
        )
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        if self.non_linear == "tanh":
            q_conv_out_tanh = Tanh(q_conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            a_conv_out_tanh = Tanh(a_conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            q_output = pool.pool_2d(input=q_conv_out_tanh, ws=self.pool_size, ignore_border=True) # max
            a_output = pool.pool_2d(input=a_conv_out_tanh, ws=self.pool_size, ignore_border=True)
        elif self.non_linear == "relu":
            q_conv_out_relu = ReLU(q_conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            a_conv_out_relu = ReLU(a_conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            q_output = pool.pool_2d(input=q_conv_out_relu, ws=self.pool_size, ignore_border=True)
            a_output = pool.pool_2d(input=a_conv_out_relu, ws=self.pool_size, ignore_border=True)
        else:
            q_output = pool.pool_2d(input=q_conv_out, ws=self.pool_size, ignore_border=True)
            a_output = pool.pool_2d(input=a_conv_out, ws=self.pool_size, ignore_border=True)

        return q_output, a_output









