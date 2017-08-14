# -*- coding: utf-8 -*-
__author__ = 'Chenjun'
from gensim.models import Word2Vec
import numpy as np
import cPickle as pickle
from gensim.models.keyedvectors import KeyedVectors
from re_sampling import re_sampling
from sklearn.utils import shuffle
import  math, re


def get_w2v_from_model(model_path,emb_path):
    """
    get w2v embeddings from w2v-model.
    :param model_path: w2v-model path.
    :param emb_path: embedding file path.
    :return:
    """
    model = Word2Vec.load(model_path)
    model.wv.save_word2vec_format(emb_path, binary=False)


def clean_text(text):
    """
    split and clean the text.
    :param text: text tokennized.
    :return:
    """
    #text = re.sub("[A-Za-z0-9]", "", text)
    text = [x for x in text.split(" ") if x!='']
    return text


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
    # train
    train_data = []
    for line in open(train_data_file_path,'r'):
        [query,answer,label] = line.split("\t")
        query = clean_text(query)
        answer = clean_text(answer)
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
    # test
    test_data = []
    for line in open(test_data_file_path,'r'):
        [query,answer,label] = line.split("\t")
        query = clean_text(query)
        answer = clean_text(answer)
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
    pickle.dump(word2index, open(word2index_save_path,'w'))
    pickle.dump(np.asarray(train_data), open(index_train_data_save_path,'w'))
    pickle.dump(np.asarray(test_data), open(index_test_data_save_path, 'w'))


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
        vec, flag = word2vec(word_emb,word,dim,scale)
        index2vec[index] = vec
        unk_count += flag
    print "emb vocab size: ",len(word_emb.vocab)
    print "unknown words count: ",unk_count
    print "index2vec size: ",len(index2vec)
    pickle.dump(np.asarray(index2vec),open(index2vec_save_path,"w"))


def split(data_file_path, group_file, index_train_data_file_path, index_valid_data_file_path, train_group_file_path, valid_group_file_path, alpha=0.9):
    """
    split train data to train set and valid set.
    :param data_file_path: train data path.
    :param group_file: train group path.
    :param index_train_data_file_path: new train data saving path.
    :param index_valid_data_file_path: new valid data saving path.
    :param train_group_file_path: new train group saving path.
    :param valid_group_file_path: new train group saving path.
    :param alpha: rate to split.
    :return:
    """
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
    pickle.dump(np.asarray(train_data),open(index_train_data_file_path,"w"))
    pickle.dump(np.asarray(valid_data),open(index_valid_data_file_path,"w"))
    pickle.dump(np.asarray(train_group), open(train_group_file_path, "w"))
    pickle.dump(np.asarray(valid_group), open(valid_group_file_path, "w"))


def word2frequency(data_file_path, save_path):
    """
    get {word:frequency} for data.
    :param data_file_path: data path
    :param save_path: save path
    :return:
    """
    word2freq = {}
    #train
    for line in pickle.load(open(data_file_path,'r')):
        [query,answer,label] = line
        for q_word in query:
            if q_word not in word2freq:
                word2freq[q_word] = 1
            else:
                word2freq[q_word] += 1
        for a_word in answer:
            if a_word not in word2freq:
                word2freq[a_word] = 1
            else:
                word2freq[a_word] += 1
    print "word2freq size: ",len(word2freq)
    pickle.dump(word2freq,open(save_path, 'w'))


def data_for_max_length_backward(data_file_path,save_file_path,data_length):
    """
    change indexed-data to fixed-length data, pad backward with zeroes.
    :param data_file_path: indexed-data file path.
    :param save_file_path: fixed-length data saving path.
    :param data_length: fixed data length.
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
    pickle.dump(np.asarray(nn_data),open(save_file_path,'w'))


def data_for_max_length_forward(data_file_path,save_file_path,data_length):
    """
    change indexed-data to fixed-length data, pad forward with zeroes.
    :param data_file_path: index-data file path.
    :param save_file_path: fixed-length data save path.
    :param data_length: fixed data length.
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
    pickle.dump(np.asarray(nn_data), open(save_file_path, 'w'))


def calculate_overlap_for_nn(data_file,overlap_file,length):
    """
    calculate overlap feature for q-a indexed data.
    :param data_file: data path.
    :param overlap_file: save path.
    :param length: data max length.
    :return:
    """
    data = pickle.load(open(data_file, "r"))
    print "overlap data length: ", len(data)
    q_data = [x[0] for x in data]
    a_data = [x[1] for x in data]
    overlap = []
    for i in xrange(len(q_data)):
        q_overlap = [0.0] * length
        a_overlap = [0.0] * length
        for j in xrange(length):
            if q_data[i][j] in a_data[i] and q_data[i][j]!=0:
                q_overlap[j] = 1
        for j in xrange(length):
            if a_data[i][j] in q_data[i] and a_data[i][j]!=0:
                a_overlap[j] = 1
        overlap.append([q_overlap,a_overlap])
    print "Overlap file(0,1) saved to " + overlap_file
    pickle.dump(np.asarray(overlap, dtype='float32'), open(overlap_file, "w"))


def calculate_lcs_for_nn(data_file,lcs_file,length):
    """
    calculate lcs feature for indexed-data.
    :param data_file: data file path
    :param lcs_file: lcs feature save path
    :param length: max data length.
    :return:
    """
    data = pickle.load(open(data_file,"r"))
    print "lcs data length: ",len(data)
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
        lcs.append([res,res])
    print "LCS file(0,1) saved to " + lcs_file
    pickle.dump(np.asarray(lcs, dtype='float32'), open(lcs_file, "w"))


def getLcs(flag, a, i, j, res):
    if i==0 or j==0:
        return
    if flag[i][j] == 'ok':
        getLcs(flag, a, i - 1, j - 1, res)
        res[i-1] = 1.
    elif flag[i][j] == 'left':
        getLcs(flag, a, i, j - 1, res)
    else:
        getLcs(flag, a, i - 1, j, res)


def question2index(word2index, question_file, save_file):
    word2index = pickle.load(open(word2index,"r"))
    question_words = [x.strip() for x in open(question_file,"r")]
    question_index = []
    for word in question_words:
        if word in word2index:
            question_index.append(word2index[word])
    pickle.dump(np.asarray(question_index),open(save_file,"w"))


def calculate_question_for_nn(data_file, question_file, length, save_file, window=3):
    """
    calculate question window feature for data.
    :param data_file: data file path
    :param question_file: question words file
    :param length: max data length
    :param save_file: save path
    :param window: question window size
    :return:
    """
    data = pickle.load(open(data_file,"r"))
    print "question data length: ", len(data)
    question_words = pickle.load(open(question_file,"r"))
    question = []
    for i in xrange(len(data)):
        q_data = data[i][0]
        a_data = data[i][1]
        q_question = [0.] * length
        a_question = [0.] * length
        for query_index in xrange(len(q_data)):
            query_word = q_data[query_index]
            if query_word in question_words and query_word!=0:
                if query_index < window and query_index + window < len(q_data):
                    query_window = q_data[0: query_index + window + 1]
                elif query_index < window and query_index + window >= len(q_data):
                    query_window = q_data[0: len(q_data)]
                elif query_index > window and query_index + window >= len(q_data):
                    query_window = q_data[query_index - window: len(q_data)]
                else:
                    query_window = q_data[query_index - window: query_index + window + 1]
                q_question[query_index] = 1.  ###
                for answer_index in xrange(len(a_data)):
                    answer_word = a_data[answer_index]
                    if answer_word in query_window and answer_word!=0:
                        a_question[answer_index] = 1.
        question.append([q_question, a_question])
    print "question file(0,1) saved to " + save_file
    pickle.dump(np.asarray(question, dtype='float32'), open(save_file, "w"))


def data_process(path, vec_dim, scale):
    """
    data process.
    :param vec_dim: vec dimension
    :param scale: random value scale
    :return:
    """
    train_data_file_path = path + "data/dbqa-data-train.txt"
    test_data_file_path = path + "data/dbqa-data-test.txt"
    emb_file_path = path + "embedding/wiki-w2v-" + str(vec_dim) + ".emb"
    print "--- index_data ---"
    #word2index index2vec index_data
    word2index_file_path = path+"pkl/word2index.pkl"
    index2vec_file_path = path+"pkl/index2vec.pkl"
    index_train_data_file_path1 = path+"pkl/data-train-all.pkl"
    index_train_data_file_path2 = path+"pkl/data-train.pkl"
    index_valid_data_file_path = path + "pkl/data-valid.pkl"
    index_test_data_file_path = path+"pkl/data-test.pkl"
    train_group_file = path+"data/dbqa-data-train.txt.group"
    train_group = path+"pkl/train_group.pkl"
    valid_group = path+"pkl/valid_group.pkl"
    sentence2index(train_data_file_path,test_data_file_path,word2index_file_path,index_train_data_file_path1,index_test_data_file_path)
    index2vector(word2index_file_path,emb_file_path,index2vec_file_path,vec_dim,scale)
    print "--- split_data ---"
    split(index_train_data_file_path1,train_group_file,index_train_data_file_path2,index_valid_data_file_path,train_group,valid_group,alpha=0.9)


if __name__ == "__main__":
    vec_dim = 50
    scale = 0.1
    length = 50
    alpha = 0.3  # sampling rate

    path = "/Users/chenjun/PycharmProjects/DBQA/"
    
    data_process(path, vec_dim=vec_dim, scale=scale)

    print "--- pad zero ---"
    data_for_max_length_forward(path+"pkl/data-train.pkl",path+"pkl/data-train-nn.pkl",length)
    data_for_max_length_forward(path + "pkl/data-valid.pkl", path + "pkl/data-valid-nn.pkl", length)
    data_for_max_length_forward(path+"pkl/data-test.pkl", path+"pkl/data-test-nn.pkl", length)

    print "--- re_sampling ---"
    sampling_data = re_sampling(pickle.load(open(path + "pkl/data-train-nn.pkl", "r")), alpha=alpha)
    pickle.dump(sampling_data, open(path + "pkl/data-train-nn.pkl", "w"))
    
    print "--- nn overlap ---"
    calculate_overlap_for_nn(path + "pkl/data-train-nn.pkl", path + "pkl/overlap01-train.pkl", length)
    calculate_overlap_for_nn(path + "pkl/data-valid-nn.pkl", path + "pkl/overlap01-valid.pkl", length)
    calculate_overlap_for_nn(path + "pkl/data-test-nn.pkl", path + "pkl/overlap01-test.pkl", length)
    
    print "--- nn lcs ---"
    calculate_lcs_for_nn(path + "pkl/data-train-nn.pkl", path + "pkl/lcs01-train.pkl", length)
    calculate_lcs_for_nn(path + "pkl/data-valid-nn.pkl", path + "pkl/lcs01-valid.pkl", length)
    calculate_lcs_for_nn(path + "pkl/data-test-nn.pkl", path + "pkl/lcs01-test.pkl", length)
    
    print "--- nn question ---"
    question2index(path+"pkl/word2index.pkl", path+"data/question_token.txt", path+"pkl/question_token.pkl")
    calculate_question_for_nn(path + "pkl/data-train-nn.pkl", path+"pkl/question_token.pkl",length,path + "pkl/question01-train.pkl", window=2)
    calculate_question_for_nn(path + "pkl/data-valid-nn.pkl", path+"pkl/question_token.pkl",length,path + "pkl/question01-valid.pkl", window=2)
    calculate_question_for_nn(path + "pkl/data-test-nn.pkl", path+"pkl/question_token.pkl",length,path + "pkl/question01-test.pkl", window=2)
