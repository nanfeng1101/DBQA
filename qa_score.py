#-*- coding:utf-8 -*-
__author__ = "ChenJun"


def rank_group(group, reverse=True):
    """
    rank QA group data by score
    :param group: QA group data
    :param reverse: True or False
    :return:
    """
    group.sort(key=lambda x: x['score'], reverse=reverse)
    return group


def group_mrr(group, label=1):
    """
    calculate @mrr for group
    :param group: QA group data [score,label]
    :param label: label
    :return:
    """
    _group_mrr = 0.0
    length = len(group)
    for index in xrange(length):
        if group[index]['label'] == label:
            _group_mrr = 1.0 / (index+1)
            break
    return _group_mrr


def group_map(group, label=1):
    """
    calculate @map for group
    :param group: QA group data [score,label]
    :param label: label
    :return:
    """
    _group_map = 0.0
    rel = 1.0
    for index in xrange(len(group)):
        if group[index]['label'] == label:
            _group_map += rel / (index+1)
            rel += 1.0
    if rel - 1 == 0.:
        return _group_map
    return _group_map / (rel-1)


def evaluate_mrr(score_list, label_list, group_list, label):
    """
    evaluate q-a by mrr
    :param score_list: score
    :param label_list: label
    :param group_list: group
    :param label: label [1,0]
    :return: a
    """
    step = 0
    eval = 0
    group_num = len(group_list)
    for i in xrange(group_num):
        group_length = group_list[i]
        group = []
        for j in xrange(step,step+group_length):
            group.append({"score":score_list[j],"label":label_list[j]})
        step += group_length
        group_ranked = rank_group(group)
        eval += group_mrr(group_ranked,label)
    eval /= group_num
    return eval


def evaluate_map(score_list, label_list, group_list, label):
    """
    evaluate q-a by map
    :param score_list: score
    :param label_list: label
    :param group_list: group
    :param label: label [1,0]
    :return:
    """
    step = 0
    eval = 0
    group_num = len(group_list)
    for i in xrange(group_num):
        group_length = group_list[i]
        group = []
        for j in xrange(step,step+group_length):
            group.append({"score":score_list[j],"label":label_list[j]})
        step += group_length
        group_ranked = rank_group(group)
        eval += group_map(group_ranked,label)
    eval /= group_num
    return eval


def qa_evaluate(score_list, label_list, group_list, label, mod="mrr"):
    """
    evaluate q-a.
    :param score_list: score
    :param label_list: label
    :param group_list: group
    :param label: label
    :param mod: mrr or map.
    :return:
    """
    eval = 0.
    if mod == "mrr":
        eval = evaluate_mrr(score_list, label_list, group_list, label)
    elif mod == "map":
        eval = evaluate_map(score_list, label_list, group_list, label)
    else:
        print ("evaluate model: %s not supported ...") % (mod)
    return eval
