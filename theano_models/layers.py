# -*- coding: utf-8 -*-
__author__ = 'Chenjun'

import theano
import theano.tensor as T
import numpy as np
import theano.tensor.shared_randomstreams


def ReLU(x):
    y = T.maximum(0.0,x)
    return (y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return (y)


def Tanh(x):
    y = T.tanh(x)
    return (y)


def Iden(x):
    y = x
    return (y)


def Dropout(rng, x, p):
    """
    p is the probablity of dropping a unit
    """
    #if p < 0. or p > 1. :
        #raise Exception('Dropout rate must be in interval [0, 1].')
    # p=1-p because 1's indicate keep and p is prob of dropping
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    sample = srng.binomial(n=1, p=1-p, size=x.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = x * T.cast(sample, theano.config.floatX)
    return output


#  rng 1234
class BatchNormLayer(object):
    """
    batch normalization.
    """
    def __init__(self, n_in, inputs):
        self.gamma = theano.shared(value=np.ones((n_in,),dtype=theano.config.floatX), name="gamma", borrow=True)
        self.beta = theano.shared(value=np.zeros((n_in,),dtype=theano.config.floatX), name="beta", borrow=True)
        mean = inputs.mean(1, keepdims=True)
        std = inputs.std(1, keepdims=True)
        std = T.sqrt(std**2 + 1e-6)  # for stability
        self.out = T.nnet.bn.batch_normalization(inputs=inputs, gamma=self.gamma, beta=self.beta, mean=mean, std=std)
        self.params = [self.gamma, self.beta]


class InteractLayer(object):
    """
    interact for question_vec and answer_vec.
    math: (q * W) * a
    q * W: [batch_size, n_q] * [n_q, dim, n_a] -> [batch_size, dim, n_a]
    (q * W) * a: [batch_size, dim, n_a] * [batch_size, n_a, 1] -> [batch_size, dim, 1]
    out: [batch_size, dim, 1] -> [batch_size, dim]
    """
    def __init__(self, rng, n_q, n_a, dim=1):
        self.dim = dim
        self.W = theano.shared(value=(rng.randn(n_q, self.dim, n_a) * 0.05).astype(theano.config.floatX), name="W", borrow=True)
        self.params = [self.W]

    def __call__(self, q, a):
        return T.batched_dot(T.tensordot(q, self.W, axes=[1, 0]), a)


class HiddenLayer(object):
    """
    hidden layer
    math: activation(W * X + b)
    """
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        self.input = input
        w_value = np.asarray(rng.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),high=np.sqrt(6.0 / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
        if activation == T.nnet.sigmoid:
            w_value *= 4
        self.W = theano.shared(value=w_value, name="W", borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name="b", borrow=True)
        self.params = [self.W, self.b]

        if activation == None:
            output = T.dot(input, self.W) + self.b
        else:
            output = activation(T.dot(input, self.W) + self.b)
        self.output = output


class DropoutHiddenLayer(HiddenLayer):
    """
    dropout after hidden layer.
    """
    def __init__(self, rng, input, n_in, n_out, activation, dropout_rate):
        super(DropoutHiddenLayer, self).__init__(rng=rng, input=input, n_in=n_in, n_out=n_out, activation=activation)
        self.output = Dropout(rng, self.output, p=dropout_rate)


class LogisticRegression(object):
    """
    logistic regression layer.
    math: softmax(W * X + b)
    """
    def __init__(self,input,n_in,n_out):
        """
        Initialize the parameters of the logistic regression
        :param input: symbolic variable that describes the input of the architecture (one minibatch)
        :param n_in: number of input units, the dimension of the space in which the data points lie
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name="W", borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name="b", borrow=True)
        self.input = input
        self.params = [self.W,self.b]
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        y.shape[0] is (symbolically) the number of rows in y, i.e.
        :param y: corresponds to a vector that gives for each example the correct label
        :return:
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def cross_entropy(self, y):
        # cross-entropy loss
        return T.mean((T.nnet.categorical_crossentropy(self.p_y_given_x, y)))

    def errors(self, y):
        """
        Return a float representing the number of errors in the mini_batch
        over the total number of examples of the mini_batch ; zero one
        loss over the size of the mini_batch
        :param y: corresponds to a vector that gives for each example the correct label
        :return:
        """
        if y.dtype[:3] == "int":
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError

    def pred_prob(self):
        return self.p_y_given_x[:, 1]


class BiLogisticRegression(object):
    """
    Bilinear Formed Logistic Regression Class
    """

    def __init__(self, linp, rinp, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type linp: theano.tensor.TensorType
        :param linp: symbolic variable that describes the left input of the
        architecture (one minibatch)

        :type rinp: theano.tensor.TensorType
        :param rinp: symbolic variable that describes the right input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of left input units

        :type n_out: int
        :param n_out: number of right input units

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=np.zeros((n_in, n_out),
                               dtype=theano.config.floatX),  # not sure should randomize the weights or not
                name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(value=0., name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.sigmoid(T.dot(T.dot(linp, self.W), rinp.T).diagonal() + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # parameters of the model
        self.params = [self.W, self.b]

    def predict(self, ldata, rdata):
        p_y_given_x = T.nnet.sigmoid(T.dot(T.dot(ldata, self.W), rdata.T).diagonal() + self.b)
        return p_y_given_x

    def get_cost(self, y):
        # cross-entropy loss
        loss = - T.mean(y * T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x))
        return loss

    def error(self, y):
        return T.mean(T.neq(self.y_pred,y))


class MLP(object):
    """
    mlp(
        hidden layer
        logistic regression layer
        )
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(input=input, rng=rng, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.logistic_regression_layer = LogisticRegression(input=self.hidden_layer.output,n_in=n_hidden,n_out=n_out)
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            abs(self.hidden_layer.W).sum()
            + abs(self.logistic_regression_layer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            (self.hidden_layer.W ** 2).sum()
            + (self.logistic_regression_layer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logistic_regression_layer.negative_log_likelihood
        )
        self.cross_entropy = (
            self.logistic_regression_layer.cross_entropy
        )
        # same holds for the function computing the number of errors
        self.errors = self.logistic_regression_layer.errors
        self.pred_prob = self.logistic_regression_layer.pred_prob
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hidden_layer.params + self.logistic_regression_layer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


class MLPDropout(object):
    """
    dropout mlp(
        hidden layer
        dropout
        logistic regression layer
        )
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_rate, activation=T.tanh):
        self.drop_hidden_layer = DropoutHiddenLayer(input=input, rng=rng, n_in=n_in, n_out=n_hidden, activation=activation,dropout_rate=dropout_rate)
        self.logistic_regression_layer = LogisticRegression(input=self.drop_hidden_layer.output, n_in=n_hidden,
                                                            n_out=n_out)
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            abs(self.drop_hidden_layer.W).sum()
            + abs(self.logistic_regression_layer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            (self.drop_hidden_layer.W ** 2).sum()
            + (self.logistic_regression_layer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logistic_regression_layer.negative_log_likelihood
        )
        self.cross_entropy = (
            self.logistic_regression_layer.cross_entropy
        )
        # same holds for the function computing the number of errors
        self.errors = self.logistic_regression_layer.errors
        self.pred_prob = self.logistic_regression_layer.pred_prob
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.drop_hidden_layer.params + self.logistic_regression_layer.params
        # end-snippet-3

        # keep track of model input
        self.input = input
