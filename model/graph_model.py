from re import S
import torch
import os
import math
import torch.nn as nn
from model.word_att_model import WordAttNet_GRU
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import numpy as np


class Sentence_Model(nn.Module):
    def __init__(self, word_hidden_size, gcn_layers,heads,num_hidden, batch_size, num_classes, dropout,pretrained_word2vec_path,
                 max_sent_length, max_word_length,n_feat,opt):
        super(Sentence_Model, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.num_hidden=num_hidden
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet_GRU(pretrained_word2vec_path, word_hidden_size)
        self.gru = nn.GRU(input_size=word_hidden_size*2, hidden_size=64,
                              bidirectional=True, batch_first=True, num_layers=1)
        self.num_classes=num_classes
        self.dropout=dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.opt=opt
       

    def forward(self, input):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.cuda_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, _ = self.word_att_net(i.permute(1, 0))
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output=torch.transpose(output,0,1)  
        output,_=self.gru(output)

        return output

class Session_Model_with_time(nn.Module):
    def __init__(self, word_hidden_size, gcn_layers,heads,num_hidden, batch_size, num_classes, dropout,pretrained_word2vec_path,
                 max_sent_length, max_word_length,n_feat,opt):
        super(Session_Model_with_time, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.num_hidden=num_hidden
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.num_classes=num_classes
        self.dropout=dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.opt=opt

        self.sentence_model=Sentence_Model(word_hidden_size, gcn_layers,heads,num_hidden, batch_size, num_classes, dropout,pretrained_word2vec_path,
                 max_sent_length, max_word_length,n_feat,opt)
        self.gat=GATConv(in_channels=128, out_channels=128)


        self.sentence_model_2=Sentence_Model(word_hidden_size, gcn_layers,heads,num_hidden, batch_size, num_classes, dropout,pretrained_word2vec_path,
                 max_sent_length, max_word_length,n_feat,opt)

        self.final_hidden_size=64
        self.num_mlps=2
        layers = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(self.final_hidden_size, self.num_classes)


        layers_time = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers_time += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs_time = nn.Sequential(*layers_time)
        self.fc_final_time = nn.Linear(self.final_hidden_size, self.num_classes)


        layers_node = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers_node += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs_node = nn.Sequential(*layers_node)
        self.fc_final_node = nn.Linear(self.final_hidden_size, self.num_classes)


        layers_comment = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers_comment += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs_comment = nn.Sequential(*layers_comment)
        self.fc_final_comment = nn.Linear(self.final_hidden_size, self.num_classes)

        

    def attention_net(self, x, query, mask=None):      
        d_k = query.size(-1)                                              
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k) 
        p_attn = F.softmax(scores, dim = -1)                              
        context = torch.matmul(p_attn, x).sum(1)       
        return context, p_attn

    def forward(self, data):   
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.cuda_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.x=torch.reshape(data.x,((len(data.y),-1,20)))
        output=self.sentence_model(data.x)



        comment_output=self.sentence_model_2(data.x)
        comment_x=self.dropout_layer(comment_output)
        comment_x = self.fcs_comment(comment_x)
        comment_logits = self.fc_final_comment(comment_x)


        tmp = torch.empty(output.shape[0], output.shape[1],output.shape[1])  
        adj = torch.ones_like(tmp).to(device)   

        output=torch.reshape(output,((-1,128)))
        data.x=output
        data.edge_index=data.edge_index.to(device)
        session_node = self.gat(data.x,data.edge_index)
        session_node = F.relu(session_node)

        data.batch=data.batch.to(device)
        gat_out = global_mean_pool(session_node,data.batch)
        session_node=torch.reshape(session_node,((len(data.y),-1,128)))
        node_feature_final=session_node
        
        x = self.dropout_layer(gat_out)
        x = self.fcs(x)
        output = self.fc_final(x)



        node_x=self.dropout_layer(node_feature_final)
        node_x = self.fcs_node(node_x)
        node_logits = self.fc_final_node(node_x)


        attn_output, attention = self.attention_net(comment_output,comment_output)
        attn_output = self.dropout_layer(attn_output)
        attn_output = self.fcs_time(attn_output)
        attn_logits = self.fc_final_time(attn_output)

        
        return output,node_logits,comment_logits,attn_logits

class Session_Model(nn.Module):
    def __init__(self, word_hidden_size, gcn_layers,heads,num_hidden, batch_size, num_classes, dropout,pretrained_word2vec_path,
                 max_sent_length, max_word_length,n_feat,opt):
        super(Session_Model, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.num_hidden=num_hidden
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length

        self.num_classes=num_classes
        self.dropout=dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.opt=opt

        self.sentence_model=Sentence_Model(word_hidden_size, gcn_layers,heads,num_hidden, batch_size, num_classes, dropout,pretrained_word2vec_path,
                 max_sent_length, max_word_length,n_feat,opt)
        self.gat=GATConv(in_channels=128, out_channels=128)

        self.final_hidden_size=64
        self.num_mlps=2
        layers = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(self.final_hidden_size, self.num_classes)


        layers_node = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers_node += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs_node = nn.Sequential(*layers_node)
        self.fc_final_node = nn.Linear(self.final_hidden_size, self.num_classes)


        layers_comment = [
            nn.Linear(128, self.final_hidden_size), nn.ReLU()]
        for _ in range(self.num_mlps - 1):
            layers_comment += [nn.Linear(self.final_hidden_size,
                                 self.final_hidden_size), nn.ReLU()]
        self.fcs_comment = nn.Sequential(*layers_comment)
        self.fc_final_comment = nn.Linear(self.final_hidden_size, self.num_classes)



    def forward(self, data):   
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.cuda_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.x=torch.reshape(data.x,((len(data.y),-1,20)))
        output=self.sentence_model(data.x)


        comment_output=output
        comment_x=self.dropout_layer(comment_output)
        comment_x = self.fcs_comment(comment_x)
        comment_logits = self.fc_final_comment(comment_x)


        tmp = torch.empty(output.shape[0], output.shape[1],output.shape[1])  
        adj = torch.ones_like(tmp).to(device)    

        output=torch.reshape(output,((-1,128)))
        data.x=output
        data.edge_index=data.edge_index.to(device)
        session_node = self.gat(data.x,data.edge_index)
        session_node = F.relu(session_node)

        data.batch=data.batch.to(device)
        gat_out = global_mean_pool(session_node,data.batch)
        session_node=torch.reshape(session_node,((len(data.y),-1,128)))
        node_feature_final=session_node
        
        
        x = self.dropout_layer(gat_out)
        x = self.fcs(x)
        output = self.fc_final(x)

        node_x=self.dropout_layer(node_feature_final)
        node_x = self.fcs_node(node_x)
        node_logits = self.fc_final_node(node_x)

        return output,node_logits,comment_logits
