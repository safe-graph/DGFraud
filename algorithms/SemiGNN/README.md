
# SemiGNN

## Paper
The SemiGNN model is proposed by the [paper](https://arxiv.org/pdf/2003.01171) below:
```bibtex
@inproceedings{wang2019semi,
  title={A Semi-supervised Graph Attentive Network for Financial Fraud Detection},
  author={Wang, Daixin and Lin, Jianbin and Cui, Peng and Jia, Quanhui and Wang, Zhen and Fang, Yanming and Yu, Quan and Zhou, Jun and Yang, Shuang and Qi, Yuan},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  pages={598--607},
  year={2019},
  organization={IEEE}
}
```


## Brief Introduction

SemiGNN takes a multi-view heterogeneous graph as input. It employs the attention mechanism to aggregate the embeddings from each view. It adds a structure-based loss with negative sampling.  

## Input Format

The input is a heterogeneous graph. We use a small example graph in our toolbox. You can find the example graph structure in **load_example_semi** function in `\utils\dataloader.py`. If you want to use your own graph as the input, just follow the same format like the example graph. 

## TODO List

- The memory-efficient implementation.
- Testing large-scale graphs. 

