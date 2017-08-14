#-*- coding:utf-8 -*-
__author__ = "ChenJun"

import jieba
import os,string
import shutil,codecs
from sklearn.utils import shuffle
import re


def get_stop_words():
    """
    load stop_words
    :return:
    """
    stopWords = []
    f = open('/Users/chenjun/PycharmProjects/DBQA/data/stopwords.txt', 'r')
    for line in f:
        if line.strip():
            stopWords.append(line.strip().decode("gbk"))
    return stopWords


def tokenzier(text, stop_words):
    """
    jieba tokenzier
    :param text: sentence
    :return:
    """
    #text = re.sub("[\s+\.\!\/_,:{}()<>?[]\;$%^*(+\"\']+|[+——！，。？：‘’”“【】{}（）《》；·～〈〉-`、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), text.decode('utf8'))
    text_list = [x.strip() for x in list(jieba.cut(text.strip(), cut_all=False)) if x not in stop_words and x!=""]
    text = " ".join(text_list)
    #text = re.sub("[A-Za-z]+", "ENG", text)
    #text = re.sub("[0-9.]+", "NUM", text)
    return text


def load_data(in_file_path,out_data_file_path):
    """
    load data/collection file from dbqa-data with tokenzier
    :param in_file_path:  file path
    :param out_data_file_path: data file path
    :return:
    """
    in_file = open(in_file_path,'r')
    out_file1 = open(out_data_file_path,'w')
    stop_words = get_stop_words()
    for line_text in in_file:
        content = line_text.split('\t')
        question = tokenzier(content[0], stop_words)
        answer = tokenzier(content[1], stop_words)
        label = content[2]
        out_file1.write((question+"\t"+answer+"\t"+label).encode('utf8'))
    out_file1.close()
    print "load data success..."


def load_data_2017(in_file_path,label_file_path,out_data_file_path):
    """
    load data/collection file from dbqa-data with tokenzier
    :param in_file_path:  file path
    :param out_data_file_path: data file path
    :return:
    """
    in_file = open(in_file_path,'r')
    out_file1 = open(out_data_file_path,'w')
    stop_words = get_stop_words()
    labels = [x.strip() for x in open(label_file_path)]
    print "label length: ",len(labels)
    num = 0
    for line_text in in_file:
        content = line_text.split('\t')
        question = tokenzier(content[0],stop_words)
        answer = tokenzier(content[1],stop_words)
        label = labels[num]+"\n"
        num += 1
        out_file1.write((question+"\t"+answer+"\t"+label).encode('utf8'))
    out_file1.close()
    print "load data success..."


def load_wiki_data(wiki_file_path,seg_file_path):
    """
    load wiki data with tokenzier.
    :param wiki_file_path: file path
    :param seg_file_path: file path
    :return:
    """
    in_file = codecs.open(wiki_file_path,"r",encoding="utf16")
    out_file = open(seg_file_path,'wb')
    stop_words = get_stop_words()
    for read_line in in_file:
        text = tokenzier(read_line.encode('utf8'), stop_words)
        out_file.write(text.encode('utf8'))
    out_file.close()
    print "end for load wiki-data..."


def get_group(data_file):
    in_file = open(data_file, "r")
    out_file = open(data_file+".group","w")
    tmp_query = ""
    group = 0
    group_collection = []
    for read_line in in_file:
        query = read_line.split("\t")[0]
        if tmp_query == query or tmp_query == "":
            group += 1
            tmp_query = query
        else:
            group_collection.append(group)
            group = 1
            tmp_query = query
    group_collection.append(group)  # the last one
    print len(group_collection)
    for g in group_collection:
        out_file.write(str(g)+"\n")
    print "end for get group file..."


if __name__ == "__main__":
    path = "/Users/chenjun/PycharmProjects/DBQA/"
    load_data(path+"NLPCC-ICCPOL-2016-QAtask/data-dbqa/nlpcc-iccpol-2016.dbqa.training-data", path+"data/dbqa-data-train.txt")
    load_data(path+"NLPCC-ICCPOL-2016-QAtask/data-dbqa/nlpcc-iccpol-2016.dbqa.testing-data", path+"data/dbqa-data-test.txt")
    #load_data_2017(path+"NLPCC2017/nlpcc-2017.dbqa.testset", path+"NLPCC2017/NLPCC2017-DBQA.test.label-only.txt", path+"data/dbqa-data-test-2017.txt")
    #load_wiki_data(path+"wiki_data/zhwiki.sim.utf8",path+"wiki_data/wiki-seg2.txt")
    get_group(path+"data/dbqa-data-train.txt")
    get_group(path+"data/dbqa-data-test.txt")
    #get_group(path+"data/dbqa-data-test-2017.txt")



