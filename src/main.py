'''
***********************************************************************
CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- A main class for CatDive.

Version: 1.0
***********************************************************************
'''


import os
import time
import torch
import argparse
from tqdm import tqdm

from model import SASRec
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a test boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='books', type=str)
parser.add_argument('--dir', default='sasrec', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=10000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--test', default=False, type=str2bool)
parser.add_argument('--topk', default=10, type=int)
parser.add_argument('--cd_neg', default=False, type=str) # sets category-biased negative sampling and high-confidence negative sampling
parser.add_argument('--multi', default=False, type=str2bool) # uses multi-embedding if true


args = parser.parse_args()
if args.cd_neg:
    args.cd_neg = [float(i) for i in args.cd_neg.split(',')]
    args.dir = 'cb_'+str(args.cd_neg[0])+'_hc_'+str(args.cd_neg[1])
if args.multi:
    # args.hidden_units = args.hidden_units * 2
    args.dir = 'multi_'+args.dir
if not os.path.isdir('runs/'+args.dataset + '_' + args.dir):
    os.makedirs('runs/'+args.dataset + '_' + args.dir)
with open(os.path.join('runs/'+args.dataset + '_' + args.dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    
    [user_train, user_valid, user_test, usernum, itemnum, catenum, category, popularity] = dataset # loads data with category and popularity information
    num_batch = len(user_train) // args.batch_size 
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
        
    # prints data information
    print('\ndataset:', args.dataset)
    print('model:', args.dir)
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('user num:', usernum)
    print('item num:', itemnum)
    print('category num:', catenum)
    
    sampler = WarpSampler(user_train, usernum, itemnum, catenum, category, popularity, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=4, cd_neg=args.cd_neg)
    
    model = SASRec(usernum, itemnum, catenum, category, args).to(args.device) 
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    
    
    model.train() 
    
    epoch_start_idx = 1
    
    
    # train or test
    if args.test:
        for file in os.listdir('runs/'+args.dataset + '_' + args.dir):
                if file.endswith(".pth"):
                    args.state_dict_path = os.path.join('runs/'+args.dataset + '_' + args.dir, file)
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    else:
        f = open(os.path.join('runs/'+args.dataset + '_' + args.dir, 'log.txt'), 'w')
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() 
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    best = 0
    
    folder = 'runs/'+args.dataset + '_' + args.dir
    fname = 'model.pth'
    fname = fname.format(args.num_epochs)   
    
    
    # training
    p = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.test: break 
        for step in tqdm(range(num_batch)): 
            u, seq, pos, neg = sampler.next_batch() 
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            if args.multi:
                for param in model.cate_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            optimizer.step()
            
        
        print("loss in epoch {}: {}".format(epoch, loss.item())) 
        
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (HR@10: %0.4f, NDCG@10: %0.4f, div@10: %0.4f)'
                    % (epoch, T, t_test[0][0], t_test[1][0], t_test[2][0]))
            f.write('%0.4f\t%0.4f\t%0.4f'
                    % (t_test[0][0], t_test[1][0], t_test[2][0]) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
            
            if t_test[1][0] > best:
                best = t_test[1][0]
                torch.save(model.state_dict(), os.path.join(folder, fname))
                p = 0
            else:
                p += 1
            
            if p > 30:
                f.write('Done')
                break   
        
    model.load_state_dict(torch.load(os.path.join(folder, fname), map_location=torch.device(args.device)))
    model.eval()
    t_test = evaluate(model, dataset, args, test_f=True)
    print('\n----- TEST (N=[10,20]) ----- \n\n HR:\t%s \n NDCG:\t%s \n div:\t%s \n' % (t_test[0], t_test[1], t_test[2]))
    
    f.close()
    sampler.close()
