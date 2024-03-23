'''
***********************************************************************
CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: model.py
- Original SASRec model and CatDive applied model.

Version: 1.0
***********************************************************************
'''


import numpy as np
import torch


'''
PointWiseFeedForward layer for SASRec

input:
* hidden_units: number of hidden units
* dropout_rate: dropout rate
returns:
* outputs: layer output
'''
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


'''
SASRec model including application of CatDive

input:
* user_num: number of users
* item_num: number of items
* cate_num: number of categories
* args: model detail including number of hidden units, max length of sequence, and more.
'''
class SASRec(torch.nn.Module):
    
    # initialization of model
    def __init__(self, user_num, item_num, cate_num, category, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.dev = args.device

        
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        
        # I1. Multi-embedding
        if args.multi:  # initialize multi-embedding
            self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_units, padding_idx=0)
            self.cate_emb = torch.nn.Embedding(self.cate_num+2, args.hidden_units, padding_idx=0)
            self.multi_layer = torch.nn.Linear(args.hidden_units * 2, args.hidden_units)
        else:
            self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_units, padding_idx=0)
            self.cate_emb = False
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.category = category

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    # gets features
    def log2feats(self, log_seqs):
        
        if self.cate_emb: # loads multi-embedding
            seqs = torch.concat([self.item_emb(torch.LongTensor(log_seqs).to(self.dev)), self.cate_emb(torch.LongTensor(self.category[log_seqs]).to(self.dev))], axis=2)
            seqs = self.multi_layer(seqs)
            seqs *= (self.item_emb.embedding_dim) ** 0.5
        else:
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
            seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    # forward propagation of the model
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):        
        log_feats = self.log2feats(log_seqs) 

        if self.cate_emb:  # loads multi-embedding
            pos_embs = torch.concat([self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)),self.cate_emb(torch.LongTensor(self.category[pos_seqs]).to(self.dev))], axis=2)
            neg_embs = torch.concat([self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)),self.cate_emb(torch.LongTensor(self.category[neg_seqs]).to(self.dev))], axis=2)
            pos_embs = self.multi_layer(pos_embs)
            neg_embs = self.multi_layer(neg_embs)
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 

    # predicts the recommendation score between all users and given item_indices when sequence history(log_seqs) is provided
    def predict(self, user_ids, log_seqs, item_indices): 
        log_feats = self.log2feats(log_seqs) 

        final_feat = log_feats[:, -1, :]

        if self.cate_emb: # loads multi-embedding
            item_embs = torch.concat([self.item_emb(torch.LongTensor(item_indices).to(self.dev)),self.cate_emb(torch.LongTensor(self.category[item_indices]).to(self.dev))], axis=1)
            item_embs = self.multi_layer(item_embs)
        else:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1).squeeze()

        return logits 