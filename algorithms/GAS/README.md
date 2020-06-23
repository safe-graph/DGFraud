
# GAS

## Paper
The GAS model is proposed by the [paper](https://arxiv.org/abs/1908.10679) below:
```bibtex
@inproceedings{li2019spam,
  title={Spam Review Detection with Graph Convolutional Networks},
  author={Li, Ao and Qin, Zhou and Liu, Runshi and Yang, Yiqun and Li, Dong},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2703--2711},
  year={2019}
}
```


## Brief Introduction

GAS directly aggregates neighbors with different node types. 

## Input Format

The input is a heterogeneous graph. We use a small example graph in our toolbox. You can find the example graph structure in **load_example_gas** function in `\utils\dataloader.py`. If you want to use your own graph as the input, just follow the same format like the example graph. 

## TODO List

