
# GraphConsis

## Paper
The GraphConsis model is proposed by the [paper](https://arxiv.org/abs/2005.00625) below:
```bibtex
@inproceedings{liu2020alleviating,
  title={Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection},
  author={Liu, Zhiwei and Dou, Yingtong and Yu, Philip S. and Deng, Yutong and Peng, Hao},
  booktitle={Proceedings of the 43nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```


## Brief Introduction

It is revised based on the [GraphSage](https://github.com/williamleif/GraphSAGE/tree/master/graphsage) model. We support multiple relations and distance sampling as mentioned in [our paper](https://arxiv.org/pdf/2005.00625.pdf).


## Run the code
Go to `algorithms/GraphConsis/`, and run the following command in the terminal:

`python -m supervised_train --train_prefix ../../dataset/ --file_name YelpChi.mat --model graphsage_mean --sigmoid True --epochs 3 --samples_1 10 -samples_2 5 --context_dim 128 --train_perc 1. --gpu 1`

or run `supervised_train.py` in your IDEs.


## Meaning of the arguments
```
--samples_1 -samples_2: the number of samples using at difference layers
--context_dim: the dimension of context embeddings
--train_perc: the percentage of training data used to train the model, 1. represents using 80% training and 20% testing, 0.5 will use 40% as training and the same 20% as testing
```
For more information about the arguments, please refer to `supervised_train.py`.

## Note
- the major differences of GraphSage and GraphConsis are in the `neighbor_sampler.py' and 'supervised_models.py'. For neighbor_sampler, we use distance sampler that computing the consistency score and sampling probability. For supervised model, the GraphConsis model considers all the relations and learn relation vectors and attention weights for each sample.

- Before running the code, please remember unzip the given dataset. 
