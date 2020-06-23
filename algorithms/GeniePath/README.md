
# GeniePath

## Paper
The GeniePath model is proposed by the [paper](https://arxiv.org/abs/1802.00910) below:
```bibtex
@inproceedings{liu2019geniepath,
  title={Geniepath: Graph neural networks with adaptive receptive paths},
  author={Liu, Ziqi and Chen, Chaochao and Li, Longfei and Zhou, Jun and Li, Xiaolong and Song, Le and Qi, Yuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={4424--4431},
  year={2019}
}
```


## Brief Introduction

GeniePath employs LSTM to learn the layers of GCN and attention mechanism to learn the neighbor weights. 

## Input Format

The input graph is homogeneous. In our toolbox, it takes a homo-graph from DBLP dataset as the input.

## TODO List

- The performance of GeniePath on DBLP needs to be tuned. 
- The implementation of GeniePath-lazy
