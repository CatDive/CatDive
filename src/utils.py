'''
***********************************************************************
CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- Dataset preperation including negative sampling and evaluation.

Version: 1.0
***********************************************************************
'''


import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
from tqdm import tqdm
import itertools


'''
Negative Sampling of CatDive

input:
* category: category of each item
* catenum: number of categories
* size: negative sample size
* ts: a set of items that the given user has interacted with
* itemnum: number of items
* popularity: popularity of each item
returns:
* neg: a negative sample
'''
def catDive_neg(a, category, catenum, size, ts, itemnum, popularity):
    
    # I2. Category-biased Negative Sampling
    category_pop = np.array(list(Counter(list(range(0, catenum+1)) + list(category[ts])).values()), dtype=np.float32) # count number of interactions between user and each category in user history sequence
    score = category_pop[category] # get number of interaction for each item's category
    score /= sum(score) # S_c(i)^cat
    
    # I3. Adjusted Negative Sampling
    if a != 0:
        score += a * (popularity / sum(popularity)) # adds S_i^pop to final score
        
    score[ts] = 0
    score = score[1:]
    neg = random.choices(list(range(1, itemnum+1)), weights=score, k=size) # samples according to the final negative sampling score
    
    return neg

'''
Random Negative Sampling for original SASRec

input:
* size: negative sample size
* ts: a set of items that the given user has interacted with
* itemnum: number of items
returns:
* neg: a negative sample
'''
def random_neq(itemnum, ts, size):
    neg = []
    count = 0
    while count < size:
        t = random.randint(1, itemnum)
        while t in ts:
            t = random.randint(1, itemnum)
        neg.append(t)
        count += 1
    return neg


'''
Sample function of all user sequences

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* catenum: number of categories
* category: category of each item
* popularity: popularity of each item
* batch_size: size of batch
* maxlen: maximum length of user sequence
* result_queue: queue to save sampling result
* SEED: random seed
* alpha: aplha to control I3. Adjusted negative sampling
'''
def sample_function(user_train, usernum, itemnum, catenum, category, popularity, batch_size, maxlen, result_queue, SEED, alpha):
    def sample():

        user = np.random.randint(1, usernum+1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum+1)

        ts = user_train[user][:-1]
        
        if len(ts) <= maxlen:
            zeros = [0] * (maxlen-len(ts)+1)
            seq = ts[:-1]
            pos = ts[1:]
            size = len(pos)
            seq = zeros + seq
            pos = zeros + pos
            if alpha != 9:
                neg = catDive_neg(alpha, category, catenum, size, ts, itemnum, popularity)
            else:
                neg = random_neq(itemnum, ts, size)
            neg = zeros + neg
        else:
            seq = ts[-(maxlen+1):-1]
            pos = ts[-maxlen:]
            if alpha != 9:
                neg = catDive_neg(alpha, category, catenum, maxlen, ts, itemnum, popularity)
            else:
                neg = random_neq(itemnum, ts, maxlen)
   
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


'''
Wrap Sampler to get all train sequences with negative samples

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* catenum: number of categories
* category: category of each item
* popularity: popularity of each item
* batch_size: size of batch
* maxlen: maximum length of user sequence
* n_workers: number of workers to use in sampling
* alpha: aplha to control I3. Adjusted negative sampling
returns:
* user train sequences with negative samples
'''
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, catenum, category, popularity, batch_size=64, maxlen=10, n_workers=4, alpha=False):
        self.result_queue = Queue(maxsize=n_workers * 20)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      catenum,
                                                      category,
                                                      popularity,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      alpha
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


'''
Train and test data partition function

input:
* fname: file name of dataset
returns:
* train and test data with information of dataset
'''
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_test = {}
    f = open('data/%s/ratings' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_test[user] = []
        else:
            user_train[user] = User[user][:-1]
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    # loads category and popularity information of items
    category = pd.read_csv('data/'+fname+'/category', header=None)
    category = np.concatenate(([0], category.to_numpy().flatten()))
    popularity = pd.read_csv('data/'+fname+'/popularity', header=None)
    popularity = np.concatenate(([0], popularity.to_numpy().flatten()))
    
    return [user_train, user_test, usernum, itemnum, category.max(), category, popularity]


'''
Train and test data partition function

input:
* model: model to evaluate
* dataset: dataset to evaluate on
* args: model details
* test_f: true if test and false if validation
returns:
* HitRate
* Category Diversity
'''
def evaluate(model, dataset, args, test_f=False):
    [train, test, usernum, itemnum, catenum, category, diversity] = copy.deepcopy(dataset)
    
    
    test_user = 0.0
    HT = [0.0, 0.0]
    diversity = [0.0, 0.0]
    div = (args.topk * (args.topk-1)) / 2
    rec_cat_list = []

    
    if not test_f:
        users = random.sample(range(1, usernum+1), 1000)
    else:
        users = range(1, usernum+1)
    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = train[u][-args.maxlen:]
        if len(seq) != args.maxlen:
            zeros = np.zeros(args.maxlen-len(seq), dtype=np.int32)
            seq = np.concatenate((zeros, seq), dtype=np.int32)

        item_idx = list(set(range(1, itemnum+1))-set(train[u]))

        predictions = model.predict(np.array([u]), np.array([seq]), item_idx)
        
        if test_f:
            for i, k in enumerate([10, 20]):

                _, topk = torch.topk(predictions, k)
                topk = np.array(item_idx)[topk.cpu()]
                
                rank = np.where(topk == test[u])[0]
                
                if rank:
                    HT[i] += 1
                    
                if k == 5:
                    rec_cat_list.append(topk)
                    
                comb = np.array(list(itertools.combinations(category[topk], 2))).T # gets combination of all items' category
                diversity[i] += len(np.where(comb[0]!=comb[1])[0]) / ((k * (k-1)) / 2)
        
        else:
            _, topk = torch.topk(predictions, args.topk)
            topk = np.array(item_idx)[topk.cpu()]
            
            rank = np.where(topk == test[u])[0]
            
            if rank:
                HT[0] += 1
            
            comb = np.array(list(itertools.combinations(category[topk], 2))).T
            diversity[0] += len(np.where(comb[0]!=comb[1])[0]) / div
        
    test_user = len(users)


    return np.array(HT) / test_user, np.array(diversity) / test_user # coverage