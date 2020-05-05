
# Player2Vec

## Paper
The Player2Vec model is proposed by the [paper](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf) below:
```
@inproceedings{zhang2019key,
  title={Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding Framework},
  author={Zhang, Yiming and Fan, Yujie and Ye, Yanfang and Zhao, Liang and Shi, Chuan},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={549--558},
  year={2019}
}
```


## Brief Introduction

Player2Vec takes a multi-view heterogeneous graph as input, it encodes homo-graph in each view with vanilla GCN and employs the attention mechanism to aggregate the embeddings from each view.

## Input Format

The input is a heterogeneous graph. We use the DBLP dataset in this toolbox.

## TODO List

- Scalable implementation.
