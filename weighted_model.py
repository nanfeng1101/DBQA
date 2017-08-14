#-*- coding:utf-8 -*-
__author__ = 'ChenJun'
import numpy as np
import cPickle as pickle
from qa_score import qa_evaluate
import matplotlib.pyplot as plt
from sklearn import svm


def average_score(file_list):
    data = []
    for file in file_list:
        score = pickle.load(open(file,"r"))
        data.append(score)
    average_data = np.mean(np.asarray(data),axis=0)
    return average_data


def ensemble(score_file_list, label_file, group_list, ensemble_file):
    score_list = average_score(score_file_list)
    pickle.dump(score_list,open(ensemble_file,"w"))
    label_list = pickle.load(open(label_file))
    mrr = qa_evaluate(score_list, label_list, group_list, label=1, mod="mrr")
    return mrr


def evaluate(path, valid_label_file, test_label_file, valid_group_file, test_group_file):
    valid_data_overlap = pickle.load(open(path + "valid_ensemble_overlap.pkl"))
    test_data_overlap = pickle.load(open(path + "test_ensemble_overlap.pkl"))

    valid_data_lcs = pickle.load(open(path + "valid_ensemble_lcs.pkl"))
    test_data_lcs = pickle.load(open(path + "test_ensemble_lcs.pkl"))

    valid_data_question = pickle.load(open(path + "valid_ensemble_question.pkl"))
    test_data_question = pickle.load(open(path + "test_ensemble_question.pkl"))

    valid_label_list = pickle.load(open(valid_label_file))
    valid_group_list = pickle.load(open(valid_group_file))

    test_label_list = pickle.load(open(test_label_file))
    test_group_list = [int(x.strip()) for x in open(test_group_file, "r")]

    valid_score_list = []
    for i in xrange(len(valid_data_lcs)):
        score = valid_data_question[i]
        valid_score_list.append(score)
    valid_mrr = qa_evaluate(valid_score_list, valid_label_list, valid_group_list, label=1, mod="mrr")
    test_score_list = []
    for i in xrange(len(test_data_lcs)):
        score = test_data_question[i]
        test_score_list.append(score)
    test_mrr = qa_evaluate(test_score_list, test_label_list, test_group_list, label=1, mod="mrr")
    print ("question: %s-%s ") % (valid_mrr, test_mrr)

    valid_score_list = []
    for i in xrange(len(valid_data_lcs)):
        score = valid_data_lcs[i]
        valid_score_list.append(score)
    valid_mrr = qa_evaluate(valid_score_list, valid_label_list, valid_group_list, label=1, mod="mrr")
    test_score_list = []
    for i in xrange(len(test_data_lcs)):
        score = test_data_lcs[i]
        test_score_list.append(score)
    test_mrr = qa_evaluate(test_score_list, test_label_list, test_group_list, label=1, mod="mrr")
    print ("lcs: %s-%s ") % (valid_mrr, test_mrr)

    valid_score_list = []
    for i in xrange(len(valid_data_lcs)):
        score = valid_data_overlap[i]
        valid_score_list.append(score)
    valid_mrr = qa_evaluate(valid_score_list, valid_label_list, valid_group_list, label=1, mod="mrr")
    test_score_list = []
    for i in xrange(len(test_data_lcs)):
        score = test_data_overlap[i]
        test_score_list.append(score)
    test_mrr = qa_evaluate(test_score_list, test_label_list, test_group_list, label=1, mod="mrr")
    print ("overlap: %s-%s ") % (valid_mrr, test_mrr)


def weighted_model(path, valid_label_file, test_label_file, valid_group_file, test_group_file):

    valid_data_overlap = pickle.load(open(path+"valid_ensemble_overlap.pkl"))
    test_data_overlap = pickle.load(open(path+"test_ensemble_overlap.pkl"))

    valid_data_lcs = pickle.load(open(path+"valid_ensemble_lcs.pkl"))
    test_data_lcs = pickle.load(open(path+"test_ensemble_lcs.pkl"))

    valid_data_question = pickle.load(open(path+"valid_ensemble_question.pkl"))
    test_data_question = pickle.load(open(path+"test_ensemble_question.pkl"))

    valid_label_list = pickle.load(open(valid_label_file))
    valid_group_list = pickle.load(open(valid_group_file))

    test_label_list = pickle.load(open(test_label_file))
    test_group_list = [int(x.strip()) for x in open(test_group_file, "r")]
    mrr_dic = {}
    p = 0.
    while p <= 1.:
        q = 0.
        while q < 1.-p:
            valid_score_list = []
            for i in xrange(len(valid_data_lcs)):
                score = p*valid_data_question[i] + q*valid_data_lcs[i] + (1.-p-q)*valid_data_overlap[i]
                valid_score_list.append(score)
            valid_mrr = qa_evaluate(valid_score_list,valid_label_list,valid_group_list,label=1,mod="mrr")

            test_score_list = []
            for i in xrange(len(test_data_lcs)):
                score = p*test_data_question[i] + q*test_data_lcs[i] + (1.-p-q)*test_data_overlap[i]
                test_score_list.append(score)
            test_mrr = qa_evaluate(test_score_list,test_label_list,test_group_list,label=1,mod="mrr")

            mrr_dic[str(p)+"-"+str(q)+"-"+str(1-p-q)] = str(valid_mrr)+"-"+str(test_mrr)
            q += 0.1
        p += 0.1
    mrr_dic = sorted(mrr_dic.iteritems(), key=lambda x: x[1])
    res(mrr_dic)

def weighted_model_2017(path, valid_label_file, test_label_file, valid_group_file, test_group_file):

    #valid_data_overlap = pickle.load(open(path+"valid_ensemble_overlap.pkl"))
    #test_data_overlap = pickle.load(open(path+"test_ensemble_overlap.pkl"))

    valid_data_lcs = pickle.load(open(path+"valid_ensemble_lcs.pkl"))
    test_data_lcs = pickle.load(open(path+"test_ensemble_lcs.pkl"))

    valid_data_question = pickle.load(open(path+"valid_ensemble_question.pkl"))
    test_data_question = pickle.load(open(path+"test_ensemble_question.pkl"))

    valid_label_list = pickle.load(open(valid_label_file))
    valid_group_list = pickle.load(open(valid_group_file))

    test_label_list = pickle.load(open(test_label_file))
    test_group_list = [int(x.strip()) for x in open(test_group_file, "r")]
    mrr_dic = {}
    p = 0.
    while p<=1.:
        q = 1. - p
        valid_score_list = []
        for i in xrange(len(valid_data_lcs)):
            score = p*valid_data_question[i] + q*valid_data_lcs[i]
            valid_score_list.append(score)
        valid_mrr = qa_evaluate(valid_score_list,valid_label_list,valid_group_list,label=1,mod="mrr")
        #print "valid mrr: ", valid_mrr

        test_score_list = []
        for i in xrange(len(test_data_lcs)):
            score = p*test_data_question[i] + q*test_data_lcs[i]
            test_score_list.append(score)
        test_mrr = qa_evaluate(test_score_list,test_label_list,test_group_list,label=1,mod="mrr")
        #print "test mrr: ", test_mrr
        mrr_dic[str(p)+"-"+str(q)] = str(valid_mrr)+"-"+str(test_mrr)
        p += 0.1

    mrr_dic = sorted(mrr_dic.iteritems(), key=lambda x: x[1])
    res(mrr_dic)

def res(mrr_dic):
    for key in mrr_dic:
        print ("question-lcs-overlap: %s  mrr: %s ")%(key[0], key[1])

if __name__ == "__main__":
    path = "result_bn_2017/"
    weighted_model(path, path+"valid_label.pkl",path+"test_label.pkl",path+"valid_group.pkl","data/dbqa-data-test-2017.txt.group")
    evaluate(path, path+"valid_label.pkl",path+"test_label.pkl",path+"valid_group.pkl","data/dbqa-data-test-2017.txt.group")

