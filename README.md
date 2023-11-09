# CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This project is a pytorch implementation of 'CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation'.
CatDive achieves the state-of-the-art performance in category diversfied sequential recommendation, achieving the highest category diversity among all competitors without a sacrifice of accuracy.
This project provides executable source code with adjustable arguments and preprocessed datasets used in the paper.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [pandas](https://pandas.pydata.org)

## Usage
You can run a demo script 'demo.sh' to compare the performance of pretrained model of original SASRec (non-diversified sequential recommendation model) and SASRec with CatDive in Movielens-1M dataset.
The result looks as follows:
```
dataset: ml-1m
model: original
average sequence length: 164.60
user num: 6040
item num: 3706
category num: 301
100%|█████████████████████| 6040/6040 [00:11<00:00, 530.88it/s]

----- TEST (N=[10,20]) ----- 

 HR:    [0.1763245  0.28857616] 
 div:   [0.87672185 0.889397  ] 


dataset: ml-1m
model: catdive
average sequence length: 164.60
user num: 6040
item num: 3706
category num: 301
100%|█████████████████████| 6040/6040 [00:13<00:00, 454.58it/s]

----- TEST (N=[10,20]) ----- 

 HR:    [0.18311258 0.29784768] 
 div:   [0.94675129 0.94859969] 
```

You can also train the model by running 'main.py'.
There are 3 arguments you can change:
- dataset (ml-1m, books, gr-r)
- multi (true, false)
    : whether to use the proposed multi-embedding or not
- alpha (any number)
    : alpha in the final negative sampling score
For example, you can train the model for Amazon Books dataset with multi-embedding and alpha of 0.2 by following code:
```
python main.py --dataset books --multi true --alpha 0.2
```


You can test the model by running 'main.py' with the argument 'test' as 'true:
```
python main.py --dataset books --multi true --dir catdive --test true
```
Make sure you put the directory excluding the name of dataset as argument 'dir'.

## Datasets
Preprocessed data are included in the data directory.
| Name | #Users | #Items | #Interactions | #Categories | Download |
| --- | ---: | ---: | ---: | ---: |--- |
|MovieLens-1M (ml-1m)| 6,040 | 3,706 | 1,000,209 | 300 | [Link](https://grouplens.org/datasets/movielens/1m/)
|Amazon Books (books) | 8,685 | 9,053 | 1,043,391 | 26 |[Link](https://nijianmo.github.io/amazon/index.html)
|GoodReads Romance (gr-r)| 4,864 | 4,964|  608,816 | 44| [Link](https://mengtingwan.github.io/data/goodreads.html)
