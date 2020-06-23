
# GEM

## Paper
The GEM model is proposed by the [paper](https://arxiv.org/pdf/2002.12307.pdf) below:
```bibtex
@inproceedings{liu2018heterogeneous,
  title={Heterogeneous graph neural networks for malicious account detection},
  author={Liu, Ziqi and Chen, Chaochao and Yang, Xinxing and Zhou, Jun and Li, Xiaolong and Song, Le},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={2077--2085},
  year={2018}
}
```


## Brief Introduction

A heterogeneous graph neural network approach for detecting malicious accounts.

## Input Format

This model uses a device graph as input. We use a small example graph in our toolbox. You can find the example graph structure in **load_example_gem** function in `\utils\dataloader.py`. If you want to use your own graph as the input, just follow the same format like the example graph. 

## TODO List

- The log loss fuction (Eq. (7) in the paper) is not implemented. Currently we use cross-entropy loss to replace it.

