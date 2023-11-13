import os
import random
import collections
import pickle

import torch
import numpy as np
import pandas as pd
import math
from torch_geometric.loader import DataLoader

class DataLoaderSession(object):
    def __init__(self, file_path,batch_size,shuffle):
        
        self.file_path=file_path
        self.batch_size=batch_size
        self.shuffle=shuffle
        self._load_input_data()

    def _load_input_data(self):       
        with open (self.file_path,"rb") as f:
            self.data = pickle.load(f)
            print(len(self.data['session_graph_data']))
    def __len__(self):
        return len(self.data['session_graph_data'])


    def sequence_pretrain_iter(self,shuffle = False,re_index = None):
        data_list = self.data['session_graph_data']
        node_label_list=self.data['node_label']
        session_id_list=self.data['session_id']
        session_len_list=self.data['session_len']
        batch_num = math.ceil(len(data_list) / self.batch_size)



        session_loader = DataLoader(data_list, batch_size=self.batch_size)   
        session_iter = iter(session_loader) 

        for i in range(batch_num):
            node_label_batch = node_label_list[i * self.batch_size: (i + 1) * self.batch_size]
            session_id_batch=session_id_list[i * self.batch_size: (i + 1) * self.batch_size]
            session_len_batch=session_len_list[i * self.batch_size: (i + 1) * self.batch_size]
            node_label_batch = torch.tensor([item.cpu().detach().numpy() for item in node_label_batch])
            session_id_batch=torch.tensor(session_id_batch)
            session_len_batch=torch.tensor(session_len_batch)
            session_batch = next(session_iter)   
            yield session_batch,node_label_batch,session_id_batch,session_len_batch

if __name__ == '__main__':
    train_loader = DataLoaderSession("../data/vine/test.pkl", batch_size=16, shuffle=True)
    for step, (data,node_label_batch,session_id_batch,session_len_batch) in enumerate(train_loader.sequence_pretrain_iter(shuffle=True,re_index=None)):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print(node_label_batch.shape)
        print(session_id_batch.shape)
        print(session_len_batch.shape)


