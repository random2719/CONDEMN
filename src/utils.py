import torch
import sys
import csv
from sklearn import metrics
import numpy as np
import json

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'f1-macro' in list_metrics:
        output['f1-macro'] = metrics.f1_score(y_true, y_pred,average='macro')
    if 'f1-micro' in list_metrics:
        output['f1-micro'] = metrics.f1_score(y_true, y_pred,average='micro')
    if 'recall' in list_metrics:
        output['recall'] = metrics.recall_score(y_true, y_pred, pos_label=1, average='macro')
    if 'precision' in list_metrics:
        output['precision'] = metrics.precision_score(y_true, y_pred, pos_label=1, average='macro')
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def get_evaluation_test(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'f1-macro' in list_metrics:
        output['f1-macro'] = metrics.f1_score(y_true, y_pred,average='macro')
    if 'f1-micro' in list_metrics:
        output['f1-micro'] = metrics.f1_score(y_true, y_pred,average='micro')
    if 'recall' in list_metrics:
        output['recall'] = metrics.recall_score(y_true, y_pred, pos_label=1, average='macro')
    if 'precision' in list_metrics:
        output['precision'] = metrics.precision_score(y_true, y_pred, pos_label=1, average='macro')
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred, target_names=['class_0','class_1']))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    session_tokens = []
    vocab_list = []
    node_label_list=[]
    with open(data_path, 'r', encoding='utf-8')as f:
        data = json.load(f)
    for i in range(len(data)):
        session = data[i]['session_tokens']
        sent_length_list.append(len(session))   
        for sent in session:
            word_length_list.append(len(sent))  

    sorted_word_length = sorted(word_length_list)   
    sorted_sent_length = sorted(sent_length_list)   

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]









