import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation,get_evaluation_test
from model.graph_model import Session_Model_with_time
from utils.loader_graph_node_label import DataLoaderSession
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import random
import logging
import torch.nn.functional as F
import json
from utils.data_for_graph_node_label import transform_geometric_data
import math

logger = logging.getLogger(__name__)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_args(opt):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(opt))

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoches", type=int, default=1)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss_weight", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/train_vine_data_json.json")
    parser.add_argument("--test_set", type=str, default="data/test_vine_data_json.json")
    parser.add_argument("--dev_set", type=str, default="data/dev_vine_data_json.json")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/condemn")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--cuda_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234, help="Number of epoches between testing phases")
    parser.add_argument("--save_model", type=str, default="best_model.ckpt")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--topk", type=float, default=0.25)
    args = parser.parse_args()
    return args



class CONDEMNTrainer(object):
    def __init__(self,max_sent_len):
        self.max_sent_len=max_sent_len

    def _go_back_original_label(self, opt):
        """
        Adding the new node to the mask and the target is updated with the predicted label.
        :param candidate: Candidate node identifier.
        :param label: Label of candidate node.
        """
        with open(opt.train_set,'r',encoding='utf-8')as f:
            data=json.load(f)
        for i in range(len(data)):
            for j in range(len(data[i]['label_mask'])):
                if data[i]['label_mask'][j]==0:
                    data[i]['node_label'][j]=-1
        with open(opt.train_set,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False)

    def _create_labeled_target(self,node_label):
        """
        Creating a mask for labeled instances and a target for them.
        """
        self.labeled_mask_list=[]
        for i in range(node_label.shape[0]): 
            labeled_mask = torch.LongTensor([0 for node in range(self.max_sent_len)])
            for j in range(node_label.shape[1]):   
                if node_label[i][j]==0 or node_label[i][j]==1:
                    labeled_mask[j]=1       
            self.labeled_mask_list.append(labeled_mask)
        self.labeled_mask=torch.cat(self.labeled_mask_list, 0)
        self.labeled_mask=torch.reshape(self.labeled_mask,(node_label.shape[0],-1))                       
        self.labeled_target = node_label


    def _create_node_indices(self,node_label):
        self.node_indices_list=[]
        for i in range(node_label.shape[0]): 
            node_indices=[index for index in range(self.max_sent_len)]  
            node_indices = torch.LongTensor(node_indices)
            self.node_indices_list.append(node_indices)
        self.node_indices=torch.cat(self.node_indices_list, 0)
        self.node_indices=torch.reshape(self.node_indices,(node_label.shape[0],-1))
        


    def _choose_best_candidate(self, predictions, indices,session_len,topk,threshold):
        """
        Choosing the best candidate based on predictions.
        :param predictions: Scores.
        :param indices: Vector of likely labels.
        :return candidate: Node chosen.
        :return label: Label of node.
        """
        candidate_list=[]
        label_list=[]
        for i in range(self.node_indices.shape[0]):  #batch
            max_len=len(self.node_indices[i])
            if max_len<=session_len[i]:
                nodes=self.node_indices[i][self.labeled_mask[i]==0]
            elif max_len>session_len[i]:
                nodes=self.node_indices[i][0:session_len[i]][self.labeled_mask[i][0:session_len[i]]==0]
            if nodes.shape[0]==0:
                candidate_list.append([])
                label_list.append([])
                continue

            top_k=int(nodes.shape[0]*topk)+1
            if max_len<=session_len[i]:
                sub_predictions=predictions[i][self.labeled_mask[i]==0]     
            elif max_len>session_len[i]:
                sub_predictions=predictions[i][0:session_len[i]][self.labeled_mask[i][0:session_len[i]]==0]

            sub_predictions_top,candidate_top=torch.topk(sub_predictions, top_k, dim=0, largest=True, sorted=True)
            mask = torch.gt(sub_predictions_top, threshold)
            candidate_top = torch.masked_select(candidate_top,  mask)

            in_candidate_list=[]
            in_label_list=[]
            for candidate in candidate_top:
                candidate = nodes[candidate]
                in_candidate_list.append(candidate)
                label = indices[i][candidate]
                in_label_list.append(label)
            candidate_list.append(in_candidate_list)
            label_list.append(in_label_list)

        return candidate_list,label_list

    def _update_target(self, opt,candidate_list, label_list,session_id):
        """
        Adding the new node to the mask and the target is updated with the predicted label.
        :param candidate: Candidate node identifier.
        :param label: Label of candidate node.
        """
        modify_session_dict={}
        for i in range(self.node_indices.shape[0]):
            if len(candidate_list[i])==0:
                continue
            candidate_set=candidate_list[i]
            label_set=label_list[i]
            for j in range(len(candidate_set)):
                self.labeled_mask[i][candidate_set[j]] = 1   
                self.labeled_target[i][candidate_set[j]] = label_set[j] 
            modify_session_dict[session_id[i].item()]=self.labeled_target[i]
        with open(opt.train_set,'r',encoding='utf-8')as f:
            data=json.load(f)
        data_id=[]
        for k in range(len(data)):
            data_id.append(data[k]['session_id'])
            if data[k]['session_id'] in modify_session_dict:
                data[k]['node_label']=modify_session_dict[data[k]['session_id']].tolist()

        with open(opt.train_set,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False)

    
    def train_process(self,opt,training_set,training_generator,max_sent_length,max_word_length,device,dev_generator,dev_set,test_generator,test_set):
        print('Training start')
        budget_size = opt.budget
        for _ in range(budget_size):
            self.train(opt,training_set,training_generator,max_sent_length,max_word_length,dev_generator,dev_set,test_generator,test_set)
    
    def test(self,opt,model,test_generator,criterion):
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(opt.saved_path + os.sep + opt.save_model))
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for (te_batch,te_node_label_batch,te_session_id_batch,te_session_len_batch) in test_generator.sequence_pretrain_iter(shuffle=True,re_index=None):
            num_sample = len(te_batch.y)
            if torch.cuda.is_available():
                te_batch.x=te_batch.x.to(device)
                te_batch.y=te_batch.y.to(device)
                te_label=te_batch.y
                te_node_label_batch=te_node_label_batch.to(device)
            with torch.no_grad():
                te_graph_predictions,te_node_predictions,te_comment_predictions,te_attn_predictions = model(te_batch)

            te_node_label_batch=torch.flatten(te_node_label_batch,0,1)     
            te_node_predictions=torch.flatten(te_node_predictions,0,1)   
            te_comment_predictions=torch.flatten(te_comment_predictions,0,1)   


            te_graph_loss = criterion(te_graph_predictions, te_label)
            te_attn_loss = criterion(te_attn_predictions, te_label)

            te_node_loss = F.cross_entropy(te_node_predictions, te_node_label_batch,ignore_index=-1)
            te_comment_loss=F.cross_entropy(te_comment_predictions, te_node_label_batch,ignore_index=-1)
            te_loss=te_graph_loss+te_attn_loss+opt.loss_weight*(te_node_loss+te_comment_loss)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_graph_predictions.clone().cpu())
        te_loss = sum(loss_ls) / len(test_generator)
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)      
        test_metrics = get_evaluation_test(te_label, te_pred.numpy(), list_metrics=["accuracy", "f1-macro","f1-micro","recall","confusion_matrix"])

        return test_metrics
    

        
    def train(self,opt):
        set_seed(opt.seed)
        output_file = open(opt.saved_path + os.sep + "logs.txt", "a+")
        output_file.write("Model's parameters: {}".format(vars(opt)))


        train_path='data/vine/train.pkl'
        dev_path='data/vine/dev.pkl'
        test_path='data/vine/test.pkl'

        transform_geometric_data(data_path=opt.train_set,
                dict_path="data/glove.6B.50d.txt",max_length_word=20,max_length_sentences=self.max_sent_len,result_graph_path=train_path)
        transform_geometric_data(data_path=opt.dev_set,
                    dict_path="data/glove.6B.50d.txt",max_length_word=20,max_length_sentences=self.max_sent_len,result_graph_path=dev_path)
        transform_geometric_data(data_path=opt.test_set,
                    dict_path="data/glove.6B.50d.txt",max_length_word=20,max_length_sentences=self.max_sent_len,result_graph_path=test_path)


        max_word_length, max_sent_length= get_max_lengths(opt.train_set)
        train_loader = DataLoaderSession(train_path, batch_size=opt.batch_size, shuffle=False)
        dev_loader = DataLoaderSession(dev_path, batch_size=opt.batch_size, shuffle=False)
        test_loader = DataLoaderSession(test_path, batch_size=opt.batch_size, shuffle=False)
        print(len(train_loader))
        print(len(dev_loader))
        print(len(test_loader))

        print('max_word_len=',max_word_length)
        print('max_sent_len=',max_sent_length)
        
        model = Session_Model_with_time(word_hidden_size=opt.word_hidden_size, gcn_layers=2,
                       heads=2,num_hidden=32,batch_size=opt.batch_size,
                       num_classes=2,dropout=opt.dropout,
                       pretrained_word2vec_path=opt.word2vec_path,
                       max_sent_length=max_sent_length, max_word_length=max_word_length,n_feat=100,opt=opt)

        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path)
        writer = SummaryWriter()

        os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            print("true")
            model.to(device)
        print(model)

        criterion = nn.CrossEntropyLoss()

        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=opt.lr)
        best_loss = 1e5
        best_epoch = 0
    
        model.train()
        num_iter_per_epoch = int(len(train_loader)/opt.batch_size)+1
        for epoch in range(opt.num_epoches):
            for iter, (batch,node_label_batch,session_id_batch,session_len_batch) in enumerate(train_loader.sequence_pretrain_iter(shuffle=True,re_index=None)):
                batch.x=batch.x.to(device)
                batch.y=batch.y.to(device)
                label=batch.y
                node_label_batch=node_label_batch.to(device)

                optimizer.zero_grad()
                graph_predictions,node_predictions,comment_predictions,attn_predictions = model(batch)
                node_label_batch=torch.flatten(node_label_batch,0,1)     
                node_predictions=torch.flatten(node_predictions,0,1)   
                comment_predictions=torch.flatten(comment_predictions,0,1)
                graph_loss = criterion(graph_predictions, label)
                attn_loss = criterion(attn_predictions, label)

                node_loss = F.cross_entropy(node_predictions,node_label_batch,ignore_index=-1)
                comment_loss=F.cross_entropy(comment_predictions,node_label_batch,ignore_index=-1)
                loss=graph_loss+attn_loss+opt.loss_weight*(node_loss+comment_loss)
                loss.backward()
                optimizer.step()
                training_metrics = get_evaluation(label.cpu().numpy(), graph_predictions.cpu().detach().numpy(), list_metrics=["accuracy","f1-macro","f1-micro","recall","confusion_matrix"])
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}, F1-macro: {}, F1-micro: {}, Recall: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["accuracy"],training_metrics["f1-macro"],training_metrics["f1-micro"],training_metrics["recall"]))
                writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
            if epoch % opt.test_interval == 0:
                model.eval()
                loss_ls = []
                te_label_ls = []
                te_pred_ls = []
                for (te_batch,te_node_label_batch,te_session_id_batch,te_session_len_batch) in dev_loader.sequence_pretrain_iter(shuffle=True,re_index=None):
                    num_sample = len(te_batch.y)
                    if torch.cuda.is_available():
                        te_batch.x=te_batch.x.to(device)
                        te_batch.y=te_batch.y.to(device)
                        te_label=te_batch.y
                        te_node_label_batch=te_node_label_batch.to(device)
                    with torch.no_grad():
                        te_graph_predictions,te_node_predictions,te_comment_predictions,te_attn_predictions = model(te_batch)
                    te_node_label_batch=torch.flatten(te_node_label_batch,0,1)     
                    te_node_predictions=torch.flatten(te_node_predictions,0,1)   
                    te_comment_predictions=torch.flatten(te_comment_predictions,0,1)
                    te_graph_loss = criterion(te_graph_predictions, te_label)
                    te_attn_loss=criterion(te_attn_predictions, te_label)

                    te_node_loss = F.cross_entropy(te_node_predictions, te_node_label_batch,ignore_index=-1)
                    te_comment_loss=F.cross_entropy(te_comment_predictions, te_node_label_batch,ignore_index=-1)
                    te_loss=te_graph_loss+te_attn_loss+opt.loss_weight*(te_node_loss+te_comment_loss)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.append(te_graph_predictions.clone().cpu())
                te_loss = sum(loss_ls) / len(dev_loader)
                te_pred = torch.cat(te_pred_ls, 0)
                te_label = np.array(te_label_ls)
                dev_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy","f1-macro","f1-micro","recall","confusion_matrix"])
                output_file.write(
                    "Epoch: {}/{} \ndev loss: {} dev accuracy: {} \ndev confusion matrix: \n{}\n\n".format(
                        epoch + 1, opt.num_epoches,
                        te_loss,
                        dev_metrics["accuracy"],
                        dev_metrics["confusion_matrix"]))
                print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}, F1-macro: {}, F1-micro: {}, Recall: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    optimizer.param_groups[0]['lr'],
                    te_loss, dev_metrics["accuracy"],dev_metrics["f1-macro"],dev_metrics["f1-micro"],dev_metrics["recall"]))
                writer.add_scalar('Dev/Loss', te_loss, epoch)
                writer.add_scalar('Dev/Accuracy', dev_metrics["accuracy"], epoch)
                model.train()
                if te_loss + opt.es_min_delta < best_loss:
                    best_loss = te_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), opt.saved_path + os.sep + opt.save_model)

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                    break
        

        print('test_result:')
        print(self.test(opt,model,test_loader,criterion))
    
        print('update data ing............')
        model.load_state_dict(torch.load(opt.saved_path + os.sep + opt.save_model))
        model.eval()
        loss_ls = []
        for (train_batch,train_node_label_batch,train_session_id_batch,train_session_len_batch) in train_loader.sequence_pretrain_iter(shuffle=True,re_index=None):

            num_sample = len(train_batch.y)
            self._create_labeled_target(train_node_label_batch)
            self._create_node_indices(train_node_label_batch)
            if torch.cuda.is_available():
                train_batch.x=train_batch.x.to(device)
                train_batch.y=train_batch.y.to(device)
                train_label=train_batch.y
                train_node_label_batch=train_node_label_batch.to(device)
            with torch.no_grad():
                train_graph_predictions,train_node_predictions,train_comment_predictions,train_attn_predictions = model(train_batch)
            train_comment_probs=nn.Softmax(dim=-1)(train_comment_predictions)
            scores_list=[]
            prediction_indices_list=[]
            for i in range(train_node_label_batch.shape[0]):
                scores, prediction_indices = train_comment_probs[i].max(dim=1)   
                scores_list.append(scores)
                prediction_indices_list.append(prediction_indices)
            candidate_list,label_list=self._choose_best_candidate(scores_list,prediction_indices_list,train_session_len_batch,opt.topk,opt.threshold)
            self._update_target(opt,candidate_list,label_list,train_session_id_batch)
        print('update_finish')




if __name__ == "__main__":
    opt = get_args()
    max_word_length, max_sent_length= get_max_lengths(opt.train_set)
    print(max_sent_length)
    max_sent_length=135
    trainer = CONDEMNTrainer(max_sent_len=max_sent_length)
    for _ in range(5):
        trainer.train(opt)





