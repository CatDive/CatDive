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
parser.add_argument('--dataset', default='ml-1m', type=str)
parser.add_argument('--dir', default='original', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=10000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--test', default=False, type=str2bool)
parser.add_argument('--topk', default=10, type=int)
parser.add_argument('--alpha', default=9, type=float) # controls alpha of adjusted negative sampling
parser.add_argument('--multi', default=False, type=str2bool) # uses multi-embedding if true


args = parser.parse_args()
if args.alpha != 9:
    args.dir = str(args.alpha)
if args.multi:
    args.hidden_units = args.hidden_units * 2
if not os.path.isdir('runs/'+args.dataset + '_' + args.dir):
    os.makedirs('runs/'+args.dataset + '_' + args.dir)
with open(os.path.join('runs/'+args.dataset + '_' + args.dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    
    [user_train, user_test, usernum, itemnum, catenum, category, popularity] = dataset # loads data with category and popularity information
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
    
    sampler = WarpSampler(user_train, usernum, itemnum, catenum, category, popularity, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=4, alpha=args.alpha)
    
    model = SASRec(usernum, itemnum, catenum, category, args).to(args.device) 
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    
    
    model.train() 
    
    epoch_start_idx = 1
            
    if args.test:
        for file in os.listdir('runs/'+args.dataset + '_' + args.dir):
                if file.endswith(".pth"):
                    args.state_dict_path = os.path.join('runs/'+args.dataset + '_' + args.dir, file)
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        model.eval()
        t_test = evaluate(model, dataset, args, test_f=True)
        print('\n----- TEST (N=[10,20]) ----- \n\n HR:\t%s \n div:\t%s \n' % (t_test[0], t_test[1]))
    else:
        f = open(os.path.join('runs/'+args.dataset + '_' + args.dir, 'log.txt'), 'w')
    
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() 
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    best = 100000000
    
    folder = 'runs/'+args.dataset + '_' + args.dir
    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)    
    
    p = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.test: break 
        for step in tqdm(range(num_batch)): 
            u, seq, pos, neg = sampler.next_batch() 
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
        if loss < best:
            best = loss
            torch.save(model.state_dict(), os.path.join(folder, fname))
            p = 0
        else:
            p += 1
        print("loss in epoch {}: {}".format(epoch, loss.item())) 
        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            print('epoch:%d, time: %f(s), test (HR@10: %0.4f, div@10: %0.4f)'
                    % (epoch, T, t_test[0][0], t_test[1][0]))
            f.write('%0.4f\t%0.4f\t%0.4f'
                    % (t_test[0][0], t_test[1][0]) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
        if p > 30:
            f.write('Done')
            break   
    f.close()
    sampler.close()
