uthor__ = 'Chenjun'
import cPickle as pickle
import numpy as np
from sklearn.utils import shuffle


def re_sampling(data, alpha):
    p_label, n_label, p_data, n_data = sample_analysis(data)
    sampling_p_data = over_sampling(p_data,int(alpha*(n_label-p_label)))
    sampling_data = sampling_p_data + n_data
    sample_analysis(sampling_data)
    sampling_data = shuffle(sampling_data)
    print "re_sampling data(shuffled) length: ",len(sampling_data)
    return np.asarray(sampling_data)


def sample_analysis(data):
    p_label, n_label = 0, 0
    p_data, n_data = [], []
    for [q, a, l] in data:
        if l == 1:
            p_label += 1
            p_data.append([q, a, l])
        else:
            n_label += 1
            n_data.append([q, a, l])
    print "p <-> n: ", p_label, n_label
    return p_label, n_label, p_data, n_data


def over_sampling(data, num):
    sampling_data = []
    print "before sampling: ", len(data)
    for i in xrange(num):
        random = np.random.random_integers(0,len(data)-1)
        sampling_data.append(data[random])
    print "sampling: ", len(sampling_data)
    sampling_data += data
    print "end of sampling: ", len(sampling_data)
    return sampling_data

