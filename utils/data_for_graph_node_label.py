from torch_geometric.data import Data
import pickle
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import json
import random



class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()
        sentence_list,session_tokens,session_labels,edge_list=[],[],[],[]
        with open(data_path,'r',encoding='utf-8')as f:
            data=json.load(f)
        for i in range(len(data)):
            sentence_list.append(data[i]['sentence_list'])
            session_tokens.append(data[i]['session_tokens'])
            session_labels.append(data[i]['session_label'])
            edge_list.append(data[i]['edge_list'])
        #print(session_tokens[2])
        self.session_list=sentence_list
        self.session_tokens=session_tokens
        self.session_labels=session_labels
        self.edge_list=edge_list
        # self.texts = texts
        # self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.session_labels))

    def __len__(self):
        return len(self.session_labels)

    def __getitem__(self, index):
        label = self.session_labels[index]
        #text = self.texts[index]
        doc = self.session_tokens[index]
        document_encode = [[self.dict.index(word) if word in self.dict else -1 for word in sentences] for sentences in doc]
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label



def transform_geometric_data(data_path,dict_path,max_length_word,max_length_sentences,result_graph_path):
    all_sentence_list,all_session_tokens,all_session_labels,all_edge_list=[],[],[],[]
    all_node_label_list=[]
    all_id_list=[]
    all_session_len_list=[]
    with open(data_path,'r',encoding='utf-8')as f:
        data=json.load(f)
    for i in range(len(data)):
        all_sentence_list.append(data[i]['sentence_list'])
        all_session_tokens.append(data[i]['session_tokens'])
        all_session_labels.append(data[i]['session_label'])
        all_edge_list.append(data[i]['edge_list'])
        all_node_label_list.append(data[i]['node_label'])
        all_id_list.append(data[i]['session_id'])
        all_session_len_list.append(data[i]['session_len'])

    print('data_nums=',len(all_sentence_list))
    dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
    dict = [word[0] for word in dict]
    print('dict_nums=',len(dict))
    all_graph_list=[]
    all_len_list=[]

    # for statistics
    node_sum_list = []
    edge_sum_list = []
    batch_node_sum = 0
    batch_edge_sum = 0
    session_graph_data_list=[]
    node_label_list=[]
    session_id_list=[]
    session_len_list=[]
    data_dict={}

    for index,sentence_list in enumerate(all_sentence_list):
        graph_dict={}
        all_len_list.append(max_length_sentences)
        if index % 1000 == 0:
            print(index)
        
        node_label = all_node_label_list[index]
        session_id=all_id_list[index]
        session_len=all_session_len_list[index]

        doc = all_session_tokens[index]
        document_encode = [[dict.index(word) if word in dict else -1 for word in sentences] for sentences in doc]
        for sentences in document_encode:
            if len(sentences) < max_length_word:
                extended_words = [-1 for _ in range(max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < max_length_sentences:
            extended_sentences = [[-1 for _ in range(max_length_word)] for _ in
                                  range(max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:max_length_word] for sentences in document_encode][
                          :max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        edge_index = [[],[]]
        
        for edge in all_edge_list[index]:
            if edge[0]>=max_length_sentences or edge[1]>=max_length_sentences:
                continue
            edge_index[0].append(edge[0])
            edge_index[1].append(edge[1])

        #for statistics
        batch_node_sum += len(sentence_list)
        batch_edge_sum += len(edge_index[0])
        #if index % 64 == 0:
        node_sum_list.append(batch_node_sum)
        edge_sum_list.append(batch_edge_sum)
        batch_node_sum = 0
        batch_edge_sum = 0
        label=all_session_labels[index]
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        document_encode = torch.tensor(document_encode,dtype=torch.long)

        if len(node_label)<max_length_sentences:
            extended_node_label = [-1 for _ in range(max_length_sentences - len(node_label))]
            node_label.extend(extended_node_label)

        node_label = node_label[:max_length_sentences] 
        node_label=torch.tensor(node_label,dtype=torch.int64)

        session_graph_data = Data(x=document_encode, edge_attr = None, edge_index=edge_index_tensor,y=label)
        session_graph_data_list.append(session_graph_data)
        node_label_list.append(node_label)
        session_id_list.append(session_id)
        session_len_list.append(session_len)


    data_dict['session_graph_data']=session_graph_data_list
    data_dict['node_label']=node_label_list
    data_dict['session_id']=session_id_list
    data_dict['session_len']=session_len_list
    with open(result_graph_path, "wb") as f2:
        pickle.dump(data_dict, f2)


    # for statistics
    print(node_sum_list)
    print(edge_sum_list)
    print("Max batch node num",max(node_sum_list))
    print("Max batch edge num", max(edge_sum_list))

if __name__ == '__main__':

    train_path='../data/vine/train.pkl'
    dev_path='../data/vine/dev.pkl'
    test_path='../data/vine/test.pkl'

    transform_geometric_data(data_path=train_path,
                dict_path="../data/glove.6B.50d.txt",max_length_word=20,max_length_sentences=75,result_graph_path=train_path)
    transform_geometric_data(data_path=dev_path,
                dict_path="../data/glove.6B.50d.txt",max_length_word=20,max_length_sentences=75,result_graph_path=dev_path)
    transform_geometric_data(data_path=test_path,
                dict_path="../data/glove.6B.50d.txt",max_length_word=20,max_length_sentences=75,result_graph_path=test_path)