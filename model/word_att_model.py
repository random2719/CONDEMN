import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import math




class WordAttNet_GRU(nn.Module):
    def __init__(self, word2vec_path, hidden_size=25):
        super(WordAttNet_GRU, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
        dict = torch.from_numpy(np.asarray(dict, dtype=np.float32))
        
        self.embed = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size,padding_idx=0).from_pretrained(dict)
        self.embed.weight = nn.Parameter(dict, requires_grad=True)


        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size,
                              bidirectional=True, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(p=0.2)
        

    def attention_net(self, x, query, mask=None):      
        d_k = query.size(-1)                                              
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        p_attn = F.softmax(scores, dim = -1)                              
        context = torch.matmul(p_attn, x).sum(1)       
        return context, p_attn

    def forward(self, input):
        

        feature=self.embed(input)
        feature=self.dropout(feature)
        feature=torch.transpose(feature,0,1)
        feature,_=self.gru(feature)
        attn_output, attention = self.attention_net(feature,feature)        
        attn_output=attn_output.unsqueeze(0)
               
        return attn_output,_





