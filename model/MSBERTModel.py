# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 08:49:25 2023

@author: Administrator
"""
import os
os.chdir('E:/github/MSBERT')
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import pickle
import torch.optim as optim
import numpy as np
from process_data import make_train_data,make_test_data
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from LoadGNPS import pro_dataset



class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, intensity):
        self.input_ids = torch.tensor(input_ids)
        self.intensity = intensity
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.intensity[idx]

class Mask(nn.Module):
    def __init__(self,ratio,max_pred):
        super(Mask , self).__init__()
        self.ratio = ratio
        self.max_pred = max_pred
    
    def forward(self,x,intensity_):
        intensity_ = intensity_.squeeze(1)
        mask_bool_list = []
        mask_token_list =[]
        mask_pos_list = []
        for p in range(x.shape[0]):
            pos_p = torch.argsort(intensity_[p,:],descending=True)
            
            # sum_intensity = torch.sum(intensity_[p,:])
            # inten_p = torch.sort(intensity_[p,:],descending=True)[0]
            # inten_accum = torch.cumsum(inten_p,dim=0)
            # inten_bool = inten_accum < 0.9*sum_intensity
            # number = torch.sum(inten_bool)
            # if number <= 5:
            #     number = 5
            # nb.append(torch.tensor(number))
            # len_mask = int(number*self.ratio)
            # if len_mask < 1:
            #     len_mask = 1
            # if len_mask > self.max_pred:
            #     len_mask = self.max_pred
           
            len_mask = 2
            mask_pos = pos_p[torch.randperm(5)[0:len_mask]].to(device)
            mask_bool = torch.rand(x.shape[1]) > 1
            mask_bool[mask_pos] = True
            mask_bool = mask_bool.to(device)
            mask_bool_list.append(mask_bool.unsqueeze(0))
            mask_token = x[p,:][mask_pos]
            if len_mask < self.max_pred:
                n_pad = self.max_pred - len_mask
                mask_pos = torch.cat((mask_pos,torch.zeros(n_pad).long().to(device)),dim=0)
                mask_token = torch.cat((mask_token,torch.zeros(n_pad).long().to(device)),dim=0)
                mask_token_list.append(mask_token.unsqueeze(0))
                mask_pos_list.append(mask_pos.unsqueeze(0))
        mask_bool = torch.cat(mask_bool_list,dim = 0)
        mask_x = torch.masked_fill(x, mask_bool,1)
        mask_token_list = torch.cat(mask_token_list,dim=0)
        mask_pos_list = torch.cat(mask_pos_list,dim=0)
        mask_token_list = mask_token_list.long()
        mask_pos_list = mask_pos_list.long()
        
        return mask_x,mask_token_list,mask_pos_list

# =============================================================================
# class Mask(nn.Module):
#     def __init__(self,max_pred,mask_vec):
#         super(Mask, self).__init__()
#         self.max_pred = max_pred
#         self.mask_vec = mask_vec
#     
#     def forward(self,embed_tensor):
#         reduce_tensor = []
#         # mask_bool = []
#         mask_pos_list = []
#         for i in range(embed_tensor.shape[0]):
#             # mask_bool_ = torch.zeros(1,embed_tensor.shape[1])
#             mask_pos = torch.LongTensor(sorted(random.sample(range(5),self.max_pred)))
#             # mask_pos = torch.LongTensor(random.sample(range(5),2))
#             # mask_bool.append(mask_bool_)
#             reduce_tensor.append(embed_tensor[i,mask_pos,:])
#             mask_tensor = embed_tensor.clone()
#             mask_tensor[i,mask_pos,:] = self.mask_vec
#             mask_pos_list.append(mask_pos.unsqueeze(0))
#         reduce_tensor = torch.stack(reduce_tensor,dim=0)  
#         # mask_bool = torch.cat(mask_bool,dim=0)
#         mask_pos_list = torch.cat(mask_pos_list,dim=0)
#         mask_pos_list = mask_pos_list.to(device)
#         return mask_tensor,reduce_tensor,mask_pos_list
# =============================================================================
        
# =============================================================================
# class Mask(nn.Module):
#     def __init__(self,ratio,max_pred):
#         super(Mask , self).__init__()
#         self.ratio = ratio
#         self.max_pred = max_pred
#         
#     def forward(self,x):
#         mask_bool_list = []
#         mask_token_list =[]
#         mask_pos_list = []
#         
#         for p in range(x.shape[0]):
#             pos_p = torch.where(x[p,:]==0)[0]
#             if len(pos_p) == 0:
#                 pos_p = x.shape[1]
#             else:
#                 pos_p = pos_p[0]
#             len_mask = int(pos_p*self.ratio)
#             if len_mask < 1:
#                 len_mask = 1
#             if len_mask > self.max_pred:
#                 len_mask = torch.tensor(self.max_pred)
#             mask_pos = torch.randperm(pos_p)[0:len_mask].to(device)
#             mask_bool = torch.rand(x.shape[1]) > 1
#             mask_bool[mask_pos] = True
#             mask_bool = mask_bool.to(device)
#             mask_bool_list.append(mask_bool.unsqueeze(0))
#             mask_token = x[p,:][mask_pos]
#             if len_mask < self.max_pred:
#                 n_pad = self.max_pred - len_mask
#                 mask_pos = torch.cat((mask_pos,torch.zeros(n_pad).long().to(device)),dim=0)
#                 mask_token = torch.cat((mask_token,torch.zeros(n_pad).long().to(device)),dim=0)
#                 mask_token_list.append(mask_token.unsqueeze(0))
#                 mask_pos_list.append(mask_pos.unsqueeze(0))
#         mask_bool = torch.cat(mask_bool_list,dim = 0)
#         mask_x = torch.masked_fill(x, mask_bool,1)
#         mask_token_list = torch.cat(mask_token_list,dim=0)
#         mask_pos_list = torch.cat(mask_pos_list,dim=0)
#         mask_token_list = mask_token_list.long()
#         mask_pos_list = mask_pos_list.long()
#         return mask_x,mask_token_list,mask_pos_list
# =============================================================================


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)
    

class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout,max_len):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # self.position = PositionalEmbedding(d_model=embed_size,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) #+ self.position(sequence)
        return self.dropout(x)

class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # print(query.size())
        # print(key.size())
        # print(value.size())
        # s1 = scores.detach().cpu().numpy()
        # plt.figure(1)
        # sns.heatmap(s1[0,0,:,:])
                 
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        # s1 = scores.detach().cpu().numpy()
        # plt.figure(2)
        # sns.heatmap(s1[0,0,:,:])

        p_attn = F.softmax(scores, dim=-1)
        # print(p_attn.size())
        # s1 = p_attn.detach().cpu().numpy()
        # plt.figure(1)
        # sns.heatmap(s1[0,0,:,:])
        

        if dropout is not None:
            p_attn = dropout(p_attn)
        # sb = torch.matmul(p_attn, value)
        # print(sb.size())
        # sb = sb.detach().cpu().numpy()
        # plt.figure(2)
        # sns.heatmap(sb[0,0,:,:])

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # q,k,v batch*nhead*maxlen*d_k
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # x batch*nhead*maxlen*batch
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # x batch*max_len*dmodel

        return self.output_linear(x)

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x.size batch*maxlen*d_model
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden,dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # x.size batch*maxlen*d_model
        x = self.output_sublayer(x, self.feed_forward)
        # x.size batch*maxlen*d_model
        return self.dropout(x)

def get_attn_pad_mask(x):
    atten_mask1 = (x == 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    atten_mask2 = atten_mask1.transpose(-2, -1)
    atten_mask = torch.add(atten_mask1,atten_mask2)
    return atten_mask

class BERT(nn.Module):

    def __init__(self, vocab_size, hidden, n_layers, attn_heads, dropout,max_len,
                 ratio,max_pred):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden,dropout=dropout,max_len=max_len)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        # self.mask_vec = torch.rand(1,hidden).to(device)
        self.mask = Mask(ratio,max_pred)
        self.fc2 = nn.Linear(hidden, vocab_size, bias=False)
        self.activ2 = nn.GELU()
        self.linear = nn.Linear(hidden, hidden)
        self.intensity_linear = nn.Linear(1, hidden)
        # self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input_id,intensity):
        # atten_mask1 = (input_id == 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
        mask_x1,mask_token1,mask_pos1 =  self.mask(input_id,intensity) 
        mask_x2,mask_token2,mask_pos2 =  self.mask(input_id,intensity)
        # inten = intensity.squeeze(1).unsqueeze(2).repeat(1,1,self.hidden)
        inten = intensity.transpose(1,2)
        inten = self.intensity_linear(inten)
        
        output1 = self.embedding(mask_x1)
        output1 = output1 + inten
        # output1 = torch.cat((output1,inten),dim=2)
        # output1 = self.linear(output1)
        
        # output1 = torch.mul(inten,output1)
        output2 = self.embedding(mask_x2)
        output2 = output2 + inten
        # output2 = torch.cat((output2,inten),dim=2)
        # output2 = self.linear(output2)
        # output2 = torch.mul(inten,output2)
        # output size,64*100*512
        
        # atten_mask = (intensity == 0).repeat(1, intensity.size(2), 1).unsqueeze(1).to(device)
        # output1,reduce_tensor1,mask_pos1 = self.mask(embed_data)
        # output2,reduce_tensor2,mask_pos2 = self.mask(embed_data)
        
        # for transformer in self.transformer_blocks:
        #     output1 = transformer.forward(output1, atten_mask)
        # pool1 = torch.matmul(intensity,output1)
        # pool1 = pool1/intensity.shape[1]
        
        # for transformer in self.transformer_blocks:
        #     output2 = transformer.forward(output2, atten_mask)
        # pool2 = torch.matmul(intensity,output2)
        # pool2 = pool2/intensity.shape[1]
        
        # mask_pos1_ = mask_pos1[:, :, None].expand(-1, -1, self.hidden)
        # h_masked1 = torch.gather(output1, 1, mask_pos1_)
        # h_masked1 = self.activ2(self.linear(h_masked1))               
        # logits_lm1 = self.fc2(h_masked1)
        
        # mask_pos2_ = mask_pos2[:, :, None].expand(-1, -1, self.hidden) 
        # h_masked2 = torch.gather(output2, 1, mask_pos2_)              
        # h_masked2 = self.activ2(self.linear(h_masked2))               
        # logits_lm2 = self.fc2(h_masked2) 
        
        
        # atten_mask1 = get_attn_pad_mask(mask_x1)
        atten_mask1 = (mask_x1 == 0).unsqueeze(1).repeat(1, mask_x1.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:
            output1 = transformer.forward(output1, atten_mask1)
        pool1 = torch.matmul(intensity,output1)
        pool1 = pool1/intensity.shape[1]
        
        # atten_mask2 = get_attn_pad_mask(mask_x2)
        atten_mask2 = (mask_x2 == 0).unsqueeze(1).repeat(1, mask_x2.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:
            output2 = transformer.forward(output2, atten_mask2)
        pool2 = torch.matmul(intensity,output2)
        pool2 = pool2/intensity.shape[1]
        
        mask_pos1 = mask_pos1[:, :, None].expand(-1, -1, self.hidden)
        h_masked1 = torch.gather(output1, 1, mask_pos1)
        h_masked1 = self.activ2(self.linear(h_masked1))               
        logits_lm1 = self.fc2(h_masked1)
        
        mask_pos2 = mask_pos2[:, :, None].expand(-1, -1, self.hidden) 
        h_masked2 = torch.gather(output2, 1, mask_pos2)              
        h_masked2 = self.activ2(self.linear(h_masked2))               
        logits_lm2 = self.fc2(h_masked2) 
        
        return logits_lm1,mask_token1,pool1,logits_lm2,mask_token2,pool2
    
    def predict(self,input_id,intensity):
        
        # atten_mask = (intensity == 0).repeat(1, intensity.size(2), 1).unsqueeze(1).to(device)
        # for transformer in self.transformer_blocks:
        #     embed_data = transformer.forward(embed_data, atten_mask)
        # pool = torch.matmul(intensity,embed_data)
        # pool = pool/intensity.shape[1]
        
        output = self.embedding(input_id)
        inten = intensity.transpose(1,2)
        inten = self.intensity_linear(inten)
        output = output + inten
        # output = torch.cat((output,inten),dim=2)
        # output = self.linear(output)
        # inten = intensity.squeeze(1).unsqueeze(2).repeat(1,1,self.hidden)
        # output = torch.mul(inten,output)
        
        atten_mask = (input_id == 0).unsqueeze(1).repeat(1, input_id.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:                                  
            output = transformer.forward(output, atten_mask)
        
        pool = torch.matmul(intensity,output)
        pool = pool/intensity.shape[1]  
        return pool



def intensity_filter(data,ratio = 0.01,min_len = 5):
    data_new = []
    for d in data:
        d_peak = d[1]
        position = [p for p,v in enumerate(d_peak[1]) if v > ratio]
        d_intensity = d_peak[1][position]
        d_p = [d_peak[0][p] for p in position]
        if len(d_p) < min_len:
            continue
        data_new.append([d[0],[d_p,d_intensity]])
    return data_new
   
def plot_step_loss(train_loss,step=100):
    all_loss = [p for i in train_loss for p in i]
    step_loss = [all_loss[i:i+step] for i in range(0,len(all_loss),step)]
    step_loss = [np.nanmean(i) for i in step_loss]
    plt.plot(step_loss)
    plt.xlabel('steps')
    plt.ylabel('loss')



if __name__ == '__main__':
    
    maxlen = 100
    batch_size = 32
    dropout = 0
    hidden=512
    n_layers = 6
    attn_heads = 16
    ratio = 0.2
    max_pred = 3
    epochs = 4
    lr = 0.0003
    
    with open('GNPSdata/ob_train_ref.pickle', 'rb') as f:
        train_ref = pickle.load(f)
    train_ref = pro_dataset(train_ref,2,99)
    # train_ref = intensity_filter(train_ref,ratio = 0.01,min_len = 5)
    with open('GNPSdata/ob_train_query.pickle', 'rb') as f:
        train_query = pickle.load(f)
    train_query = pro_dataset(train_query,2,99)
    # train_query = intensity_filter(train_query,ratio = 0.01,min_len = 5)
    with open('GNPSdata/ob_test_ref.pickle', 'rb') as f:
        test_ref = pickle.load(f)
    test_ref = pro_dataset(test_ref,2,99)
    # test_ref = intensity_filter(test_ref,ratio = 0.01,min_len = 5)
    with open('GNPSdata/ob_test_query.pickle', 'rb') as f:
        test_query = pickle.load(f)
    test_query= pro_dataset(test_query,2,99)
    # test_query = intensity_filter(test_query,ratio = 0.01,min_len = 5)
    msms1 = [i[2] for i in train_ref]
    msms2 = [i[2] for i in train_query]
    msms3 = [i[2] for i in test_ref]
    msms4 = [i[2] for i in test_query]
    precursor1 = [i[1] for i in train_ref]
    precursor2 = [i[1] for i in train_query]
    precursor3 = [i[1] for i in test_ref]
    precursor4 = [i[1] for i in test_query]
    smiles1 = [i[0] for i in train_ref]
    smiles2 = [i[0] for i in train_query]
    smiles3 = [i[0] for i in test_ref]
    smiles4 = [i[0] for i in test_query]
    # msms = msms1+msms2+msms3+msms4
    # msms = msms1
    
    # all_data,word2idx = make_train_data(msms,maxlen)
    # train_data = all_data[0:len(msms1)]
    train_data,word2idx = make_train_data(msms1,precursor1,maxlen)
    
    vocab_size = len(word2idx)
    input_ids, intensity = zip(*train_data) 
    intensity = [torch.FloatTensor(i) for i in intensity] 
    
    model = BERT(vocab_size, hidden, n_layers, attn_heads, dropout,maxlen,ratio,max_pred)
    
    
# data1 = torch.randint(0,100,(32,10))
# intensity  = torch.rand(32,1,10)
# mask = (data1 > 0).unsqueeze(1).repeat(1, data1.size(1), 1).unsqueeze(1)
# mask = mask.numpy()
# mask2 = get_attn_pad_mask(data1,data1)
# mask2 = mask2.numpy()
# test1 = bert(data1,intensity)




# dataset_list = modelEmbed(model,msms1,batch_size,embed_dim)
# dataset_arr = np.concatenate(dataset_list)
# dataset_arr = dataset_arr.reshape(dataset_arr.shape[0],dataset_arr.shape[2])
# dataset_smiles = [i[0] for i in train_ref]

# query_list = modelEmbed(model,msms2,batch_size,embed_dim)
# query_arr = np.concatenate(query_list)
# query_arr = query_arr.reshape(query_arr.shape[0],query_arr.shape[2])
# query_smiles = [i[0] for i in train_query]
# top = search_top(dataset_arr,query_arr,dataset_smiles,query_smiles,batch=50)

# #test
# data_d = train_ref+test_ref
# msms_d = [i[1] for i in data_d]
# dataset_list = modelEmbed(model,msms_d,batch_size,embed_dim)
# dataset_arr = np.concatenate(dataset_list)
# dataset_arr = dataset_arr.reshape(dataset_arr.shape[0],dataset_arr.shape[2])
# dataset_smiles = [i[0] for i in data_d]

# query_msms = [i[1] for i in test_query]
# query_list = modelEmbed(model,query_msms,batch_size,embed_dim)
# query_arr = np.concatenate(query_list)
# query_arr = query_arr.reshape(query_arr.shape[0],query_arr.shape[2])
# query_smiles = [i[0] for i in test_query]
# top2 = search_top(dataset_arr,query_arr,dataset_smiles,query_smiles,batch=50)

    dataset_msms = make_test_data(msms1,precursor1,word2idx,maxlen)
    dataset_list = model_embed(model,dataset_msms,batch_size)
    dataset_arr = np.concatenate(dataset_list)
    dataset_arr = dataset_arr.reshape(dataset_arr.shape[0],dataset_arr.shape[2])
    dataset_smiles = [i[0] for i in train_ref]
    
    query_msms = [i[2] for i in train_query]
    query_msms = make_test_data(query_msms,precursor2,word2idx,maxlen)
    query_list = model_embed(model,query_msms,batch_size)
    query_arr = np.concatenate(query_list)
    query_arr = query_arr.reshape(query_arr.shape[0],query_arr.shape[2])
    query_smiles = [i[0] for i in train_query]
    top = search_top(dataset_arr,query_arr,dataset_smiles,query_smiles,batch=50)
    
    #test data
    query_msms = [i[2] for i in test_query]
    query_msms = make_test_data(query_msms,precursor4,word2idx,maxlen)
    query_list = model_embed(model,query_msms,batch_size)
    query_arr = np.concatenate(query_list)
    query_arr = query_arr.reshape(query_arr.shape[0],query_arr.shape[2])
    
    













