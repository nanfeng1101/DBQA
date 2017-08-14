# -*- coding: utf-8 -*-
__author__ = 'Chenjun'
from gensim.models import Word2Vec
import numpy as np
import cPickle as pickle
from gensim.models.keyedvectors import KeyedVectors
from sklearn.utils import shuffle


def get_w2v_from_model(model_path,emb_path):
    """
    get w2v embeddings from w2v-model.
    :param model_path: w2v-model path.
    :param emb_path: embedding file path.
    :return:
    """
    model = Word2Vec.load(model_path)
    model.wv.save_word2vec_format(emb_path,binary=False)


def word2vec(word_emb,word,vec_dim,scale):
    """
    word to emb-vec, if found in trained emb-file, use the existing vec, otherwise value it randomly.
    :param word_emb: trained emb-vec file.
    :param word: input
    :param vec_dim: vec dimension
    :param scale: random value scale
    :return:
    """
    word = word.decode("utf8") ###str 2 utf8
    unknown_word = np.random.uniform(-scale,scale,vec_dim)
    if word in word_emb:
        res = word_emb[word]
        flag = 0
    else:
        res = unknown_word
        flag = 1
    return res,flag


def sentence2index(train_data_file_path, test_data_file_path, word2index_save_path, index_train_data_save_path, index_test_data_save_path):
    """
    build a word2index dict, replace words in data with index and save them.
    :param train_data_file_path: train data file path
    :param test_data_file_path: test data file path
    :param word2index_save_path: word2index dict file path
    :param index_train_data_save_path: index data file path(train)
    :param index_test_data_save_path: index data file path(test)
    :return:
    """
    word2index = {}
    index = 1 # 0 used for pad 0(fill the blank).
    #train
    train_data = []
    for line in open(train_data_file_path,'r'):
        [query,answer,label] = line.split("\t")
        query = query.split(" ")
        answer = answer.split(" ")
        label = int(label)
        query_index = []
        answer_index = []
        for q_word in query:
            if q_word not in word2index:
                word2index[q_word] = index
                query_index.append(index)
                index += 1
            else:
                query_index.append(word2index[q_word])
        for a_word  in answer:
            if a_word not in word2index:
                word2index[a_word] = index
                answer_index.append(index)
                index += 1
            else:
                answer_index.append(word2index[a_word])
        train_data.append([query_index,answer_index,label])
    #test
    test_data = []
    for line in open(test_data_file_path,'r'):
        [query,answer,label] = line.split("\t")
        query = query.split(" ")
        answer = answer.split(" ")
        label = int(label)
        query_index = []
        answer_index = []
        for q_word in query:
            if q_word not in word2index:
                word2index[q_word] = index
                query_index.append(index)
                index += 1
            else:
                query_index.append(word2index[q_word])
        for a_word  in answer:
            if a_word not in word2index:
                word2index[a_word] = index
                answer_index.append(index)
                index += 1
            else:
                answer_index.append(word2index[a_word])
        test_data.append([query_index,answer_index,label])
    print "word2index size: ",len(word2index)
    print "data size: ",len(train_data), len(test_data)
    pickle.dump(word2index,open(word2index_save_path,'w'))
    pickle.dump(train_data,open(index_train_data_save_path,'w'))
    pickle.dump(test_data, open(index_test_data_save_path, 'w'))


def index2vector(word2index_file_path,emb_file_path,index2vec_save_path,dim,scale):
    """
    build the index2emb-vec matrix and save it.
    :param word2index_file_path: word2index file path
    :param emb_file_path: trained embedding file path
    :param index2vec_save_path: index2vec file path
    :param dim: vec dimension
    :param scale: random value scale
    :return:
    """
    unk_count = 0
    word2index = pickle.load(open(word2index_file_path,"r"))
    word_emb = KeyedVectors.load_word2vec_format(emb_file_path,binary=False)
    vocab_size = len(word2index)
    index2vec = np.zeros((vocab_size + 1,dim),dtype="float32")
    index2vec[0] = np.zeros(dim) # vector 0 used for words to fill the blank(pad 0).
    for word in word2index:
        index = word2index[word]
        vec,flag = word2vec(word_emb,word,dim,scale)
        index2vec[index] = vec
        unk_count += flag
    print "emb vocab size: ",len(word_emb.vocab)
    print "unknown words count: ",unk_count
    print "index2vec size: ",len(index2vec)
    pickle.dump(index2vec,open(index2vec_save_path,"w"))


def split(data_file_path, group_file, index_train_data_file_path, index_valid_data_file_path, train_group_file_path, valid_group_file_path, alpha=0.8,):
    data = pickle.load(open(data_file_path,"r"))
    group = [int(x.strip()) for x in open(group_file, "r")]
    data_collection = []
    step = 0
    for i in xrange(len(group)):
        group_length = group[i]
        group_data = []
        for j in xrange(step, step + group_length):
            group_data.append(data[j])
        step += group_length
        data_collection.append(group_data)
    print "group data length: ", len(data_collection)
    data_collection,group = shuffle(data_collection,group)
    _train_data = data_collection[:int(alpha*len(data_collection))]
    train_data = []
    for group_data in _train_data:
        for data in group_data:
            train_data.append(data)
    train_group = group[:int(alpha*len(data_collection))]
    _valid_data = data_collection[int(alpha*len(data_collection)):]
    valid_data = []
    for group_data in _valid_data:
        for data in group_data:
            valid_data.append(data)
    valid_group = group[int(alpha*len(data_collection)):]
    print "train data length: ", len(train_data)
    print "valid data length: ", len(valid_data)
    pickle.dump(train_data,open(index_train_data_file_path,"w"))
    pickle.dump(valid_data,open(index_valid_data_file_path,"w"))
    pickle.dump(train_group, open(train_group_file_path, "w"))
    pickle.dump(valid_group, open(valid_group_file_path, "w"))


def data_process(path,vec_dim,scale):
    """
    data process
    :param vec_dim: vec dimension
    :param scale: random value scale
    :return:
    """
    #word2index index2vec index_data
    train_data_file_path = path + "data/dbqa-data-train.txt"
    test_data_file_path = path + "data/dbqa-data-test.txt"
    emb_file_path = path + "save_models/wiki-w2v-" + str(vec_dim) + ".emb"
    print "--- index_data ---"
    # word2index index2vec index_data
    word2index_file_path = path + "pkl/word2index.pkl"
    index2vec_file_path = path + "pkl/index2vec.pkl"
    index_train_data_file_path = path + "pkl/data-train.pkl"
    index_valid_data_file_path = path + "pkl/data-valid.pkl"
    index_test_data_file_path = path + "pkl/data-test.pkl"
    train_group_file = path + "data/dbqa-data-train.txt.group"
    train_group = path + "pkl/train_group.pkl"
    valid_group = path + "pkl/valid_group.pkl"
    sentence2index(train_data_file_path, test_data_file_path, word2index_file_path, index_train_data_file_path,
                   index_test_data_file_path)
    index2vector(word2index_file_path, emb_file_path, index2vec_file_path, vec_dim, scale)
    print "--- split_data ---"
    split(index_train_data_file_path, train_group_file, index_train_data_file_path, index_valid_data_file_path,
          train_group, valid_group, alpha=0.8)


def data_for_max_length_backward(data_file_path,save_file_path,data_length):
    """
    change index-data to a fixed-length data, pad with zeroes.
    :param data_file_path: index-data file path
    :param save_file_path: fixed-length data path
    :param data_length: fixed data length
    :return:
    """
    print "pad backward with zero..."
    nn_data = []
    for data in pickle.load(open(data_file_path,'r')):
        q_data = data[0]
        a_data = data[1]
        label = data[2]
        if len(q_data) >= data_length:
            q_data = q_data[:data_length]
        else:
            for i in xrange(len(q_data), data_length):
                q_data.append(0)  # pad with zeros.
        if len(a_data) >= data_length:
            a_data = a_data[:data_length]
        else:
            for i in xrange(len(a_data),data_length):
                a_data.append(0)
        nn_data.append([q_data,a_data,label])
    pickle.dump(nn_data,open(save_file_path,'w'))


def data_for_max_length_forward(data_file_path,save_file_path,data_length):
    """
    change index-data to a fixed-length data, pad with zeroes.
    :param data_file_path: index-data file path
    :param save_file_path: fixed-length data path
    :param data_length: fixed data length
    :return:
    """
    print "pad forward with zero..."
    nn_data = []
    for data in pickle.load(open(data_file_path, 'r')):
        q_data = data[0]
        a_data = data[1]
        label = data[2]
        if len(q_data) >= data_length:
            q_data = q_data[:data_length]
        else:
            pad = [0] * (data_length-len(q_data))
            q_data = pad + q_data
        if len(a_data) >= data_length:
            a_data = a_data[:data_length]
        else:
            pad = [0] * (data_length - len(a_data))
            a_data = pad + a_data
        nn_data.append([q_data, a_data, label])
    pickle.dump(nn_data, open(save_file_path, 'w'))


def get_pair_data(data_file, group_file, save_path):
    data = pickle.load(open(data_file,"r"))
    group = pickle.load(open(group_file,"r"))
    pair_data = []
    step = 0
    for i in xrange(len(group)):
        group_length = group[i]
        group_data = []
        for j in xrange(step,step+group_length):
            group_data.append(data[j])
        step += group_length
        pair_data += group_pair_data(group_data)
    print "pair data length: ",len(pair_data)
    pickle.dump(pair_data, open(save_path, "w"))


def group_pair_data(group_data):
    p_data = []
    n_data = []
    pair_data = []
    for data in group_data:
        if data[2] == 1: #positive
            p_data.append(data)
        if data[2] == 0: #negative
            n_data.append(data)
    ###
    ###
    for p_d in p_data:
        for n_d in n_data:
            pair_data.append([p_d[0],p_d[1],n_d[1]])
    return pair_data  # shuffle?


def calculate_overlap_train(data_file, overlap_file, length):
    data = pickle.load(open(data_file, "r"))
    print "overlap train data length: ", len(data)
    q_data = [x[0] for x in data]
    a1_data = [x[1] for x in data]
    a2_data = [x[2] for x in data]
    overlap = []
    for i in xrange(len(q_data)):
        q1_overlap = [0.0] * length
        q2_overlap = [0.0] * length
        a1_overlap = [0.0] * length
        a2_overlap = [0.0] * length
        for j in xrange(length):
            if q_data[i][j] in a1_data[i] and q_data[i][j] != 0:
                q1_overlap[j] = 1.0
        for j in xrange(length):
            if q_data[i][j] in a2_data[i] and q_data[i][j] != 0:
                q2_overlap[j] = 1.0
        for j in xrange(length):
            if a1_data[i][j] in q_data[i] and a1_data[i][j] != 0:
                a1_overlap[j] = 1.0
        for j in xrange(length):
            if a2_data[i][j] in q_data[i] and a2_data[i][j] != 0:
                a2_overlap[j] = 1.0
        overlap.append([q1_overlap, q2_overlap, a1_overlap, a2_overlap])
    print "Overlap train file(0,1) saved to " + overlap_file
    pickle.dump(np.asarray(overlap, dtype='float32'), open(overlap_file, "w"))


def calculate_overlap_test(data_file,overlap_file,length):
    data = pickle.load(open(data_file, "r"))
    print "overlap test data length: ", len(data)
    q_data = [x[0] for x in data]
    a_data = [x[1] for x in data]
    overlap = []
    for i in xrange(len(q_data)):
        q_overlap = [0.0] * length
        a_overlap = [0.0] * length
        for j in xrange(length):
            if q_data[i][j] in a_data[i] and q_data[i][j]!=0:
                q_overlap[j] = 1.0
        for j in xrange(length):
            if a_data[i][j] in q_data[i] and a_data[i][j]!=0:
                a_overlap[j] = 1.0
        overlap.append([q_overlap,a_overlap])
    print "Overlap test file(0,1) saved to " + overlap_file
    pickle.dump(np.asarray(overlap, dtype='float32'), open(overlap_file, "w"))


def calculate_lcs_test(data_file,lcs_file,length):
    data = pickle.load(open(data_file,"r"))
    print "test lcs data length: ",len(data)
    lcs = []
    for index in xrange(len(data)):
        lcs_mat = [[0 for i in range(length + 1)] for j in range(length + 1)]
        flag = [[0 for i in range(length + 1)] for j in range(length + 1)]
        q_data = data[index][0]
        a_data = data[index][1]
        for i in xrange(length):
            for j in xrange(length):
                if q_data[i] == a_data[j] and q_data[i]!=0 and a_data[j]!=0:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i][j] + 1
                    flag[i + 1][j + 1] = 'ok'
                elif lcs_mat[i + 1][j] > lcs_mat[i][j + 1]:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i + 1][j]
                    flag[i + 1][j + 1] = 'left'
                else:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i][j + 1]
                    flag[i + 1][j + 1] = 'up'
        res = [0.] * length
        getLcs(flag, q_data, length, length, res)
        lcs.append([res, res])
    print "test LCS file(0,1) saved to " + lcs_file
    pickle.dump(np.asarray(lcs, dtype='float32'), open(lcs_file, "w"))


def calculate_lcs_train(data_file,lcs_file,length):
    data = pickle.load(open(data_file,"r"))
    print "train lcs data length: ",len(data)
    lcs = []
    for index in xrange(len(data)):
        q_data = data[index][0]
        a1_data = data[index][1]
        a2_data = data[index][2]

        lcs_mat = [[0 for i in range(length + 1)] for j in range(length + 1)]
        flag = [[0 for i in range(length + 1)] for j in range(length + 1)]
        for i in xrange(length):
            for j in xrange(length):
                if q_data[i] == a1_data[j] and q_data[i]!=0 and a1_data[j]!=0:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i][j] + 1
                    flag[i + 1][j + 1] = 'ok'
                elif lcs_mat[i + 1][j] > lcs_mat[i][j + 1]:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i + 1][j]
                    flag[i + 1][j + 1] = 'left'
                else:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i][j + 1]
                    flag[i + 1][j + 1] = 'up'
        res1 = [0.] * length
        getLcs(flag, q_data, length, length, res1)

        lcs_mat = [[0 for i in range(length + 1)] for j in range(length + 1)]
        flag = [[0 for i in range(length + 1)] for j in range(length + 1)]
        for i in xrange(length):
            for j in xrange(length):
                if q_data[i] == a2_data[j] and q_data[i] != 0 and a2_data[j] != 0:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i][j] + 1
                    flag[i + 1][j + 1] = 'ok'
                elif lcs_mat[i + 1][j] > lcs_mat[i][j + 1]:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i + 1][j]
                    flag[i + 1][j + 1] = 'left'
                else:
                    lcs_mat[i + 1][j + 1] = lcs_mat[i][j + 1]
                    flag[i + 1][j + 1] = 'up'
        res2 = [0.] * length
        getLcs(flag, q_data, length, length, res2)
        lcs.append([res1,res1,res2,res2])
    print "train LCS file(0,1) saved to " + lcs_file
    pickle.dump(np.asarray(lcs, dtype='float32'), open(lcs_file, "w"))


def getLcs(flag, a, i, j, res):
    if i == 0 or j == 0:
        return
    if flag[i][j] == 'ok':
        getLcs(flag, a, i - 1, j - 1, res)
        res[i-1] = 1.
    elif flag[i][j] == 'left':
        getLcs(flag, a, i, j - 1, res)
    else:
        getLcs(flag, a, i - 1, j, res)


if __name__ == "__main__":
    path = "/home/chenjun/project/DBQA_NN/"
    vec_dim = 50
    length = 50

    data_process(path,vec_dim=vec_dim,scale=0.1)
    data_for_max_length_forward(path+"pkl/data-train.pkl", path+"pkl/data-train-nn.pkl", length)
    data_for_max_length_forward(path + "pkl/data-valid.pkl", path + "pkl/data-valid-nn.pkl", length)
    data_for_max_length_forward(path+"pkl/data-test.pkl", path+"pkl/data-test-nn.pkl", length)

    get_pair_data(path+"pkl/data-train-nn.pkl", path+"pkl/train_group.pkl", path+"pkl/data-train-pair.pkl")

    calculate_overlap_train(path+"pkl/data-train-pair.pkl", path+"pkl/overlap01-train-pair.pkl", length)
    calculate_overlap_test(path + "pkl/data-valid-nn.pkl", path + "pkl/overlap01-valid-pair.pkl", length)
    calculate_overlap_test(path+"pkl/data-test-nn.pkl", path+"pkl/overlap01-test-pair.pkl", length)
    '''
    calculate_lcs_train(path + "pkl/data-train-pair.pkl", path + "pkl/lcs01-train-pair.pkl", length)
    calculate_lcs_test(path + "pkl/data-valid-nn.pkl", path + "pkl/lcs01-valid-pair.pkl", length)
    calculate_lcs_test(path + "pkl/data-test-nn.pkl", path + "pkl/lcs01-test-pair.pkl", length)
    '''


