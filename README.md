# CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This project is a pytorch implementation of 'CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation'.
CatDive achieves the state-of-the-art performance in category diversfied sequential recommendation, achieving the highest category diversity among all competitors without sacrificing accuracy.
This project provides executable source code with adjustable arguments and preprocessed datasets used in the paper.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [pandas](https://pandas.pydata.org)

## Usage
You can run a demo script 'demo.sh' to compare the performance of pretrained model of SASRec (non-diversified sequential recommendation model) and SASRec with CatDive in Amazon Books dataset.
The result looks as follows:
```
dataset: books
model: sasrec
average sequence length: 118.14
user num: 8685
item num: 9053
category num: 26
100%|███████████████████| 8685/8685 [00:23<00:00, 363.01it/s]

----- TEST (N=[10,20]) ----- 

 HR:    [0.1845711  0.26678181] 
 NDCG:  [0.10140844 0.12210846] 
 div:   [0.49118659 0.48891313] 


dataset: books
model: multi_catdive
average sequence length: 118.14
user num: 8685
item num: 9053
category num: 26
100%|███████████████████| 8685/8685 [00:25<00:00, 338.67it/s]

----- TEST (N=[10,20]) ----- 

 HR:    [0.2039148  0.28762234] 
 NDCG:  [0.11137709 0.13246349] 
 div:   [0.61185185 0.60768112] 
```

You can also train the model by running 'main.py'.
There are 3 arguments you can change:
- dataset (books, kindle, ml-1m)
- multi (true, false)
    : whether to use the proposed multi-embedding or not
- cd_neg (0 or 1, any number for alpha)
    : <br/> [the first number] - whether to use category-biased negative sampling or not (1 to use, 0 to not use)
      <br/> [the second number] - alpha to control the high-confidence negative sampling

For example, you can train the model for Amazon Books dataset with multi-embedding, category-biased negative sampling, and alpha of 0.2 for high-confidence negative sampling by following code:
```
python main.py --dataset books --multi true --cd_neg 1, 0.2
```


You can test the model by running 'main.py' with the argument 'test' as 'true:
```
python main.py --dataset books --multi true --dir catdive --test true
```
Make sure you put the directory excluding the name of dataset as argument 'dir'.

## Datasets
Preprocessed data are included in the data directory.
| Name | #Users | #Items | #Interactions | #Categories | Download |
| --- | ---: | ---: | ---: | ---: | :---: |
|Amazon Books (books) | 8,685 | 9,053 | 1,043,391 | 26 |[Link](https://nijianmo.github.io/amazon/index.html)
|Amazon Kindle (kindle)| 8,530 | 10,057 | 438,541 | 17 | [Link](https://nijianmo.github.io/amazon/index.html)
|MovieLens 1M (ml-1m)| 6,040 | 3,706 | 1,000,209 | 301 | [Link](https://grouplens.org/datasets/movielens/1m/)
