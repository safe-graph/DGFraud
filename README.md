<h3 align="center">
<p>Under Building Now. The first version is expected to be released in mid May, 2020.
</h3>

<p align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/194400/49531010-48dad180-f8b1-11e8-8d89-1e61320e1d82.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="http://makeapullrequest.com">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square">
    </a>
    <a href="https://github.com/safe-graph/DGFraud/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/safe-graph/DGFraud">
    </a>
    <a href="https://github.com/safe-graph/DGFraud">
        <img alt="Downloads" src="https://img.shields.io/github/downloads/safe-graph/DGFraud/total">
    </a>
    <a href="https://github.com/safe-graph/DGFraud/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/safe-graph/DGFraud">
    </a>
</p>

<h3 align="center">
<p>A Deep Graph-based Toolbox for Fraud Detection
</h3>

Introduction: **DGFraud** is a Graph Neural Network (GNN) based toolbox for fraud detection. It integrates the implementation & comparison of state-of-the-art GNN-based fraud detection models. It also includes several utility functions such as graph preprocessing, graph sampling, and performance evaluation. The introduction of implemented models can be found [here](#implemented-models). <!-- (Add introduction blogs links). -->

We welcome contributions on adding new fraud detectors and extending the features of the toolbox. Some of the planned features are listed in [TODO list](#todo-list). 

If you feel this repo is useful, please cite the [paper]() below:
```
@inproceedings{liu2020alleviating,
  title={Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection},
  author={Liu, Zhiwei and Dou, Yingtong and Yu, Philip S. and Deng, Yutong and Peng, Hao},
  booktitle={Proceedings of the 43nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

**Useful Resources**
- [Graph-based Fraud Detection Paper List](https://github.com/safe-graph/graph-fraud-detection-papers) 
- [Awesome Fraud Detection Papers](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers)
- [Attack and Defense Papers on Graph Data](https://github.com/safe-graph/graph-adversarial-learning-literature)
- [PyOD: A Python Toolbox for Scalable Outlier Detection (Anomaly Detection)](https://github.com/yzhao062/pyod)
- [PyODD: An End-to-end Outlier Detection System](https://github.com/datamllab/pyodds)
- [DGL: Deep Graph Library](https://github.com/dmlc/dgl)
- [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/)

**Table of Contents**
- [Installation](#installation)
- [User Guide](#user-guide)
- [Implemented Models](#implemented-models)
- [Model Comparison](#model-comparison)
- [TODO List](#todo-list)
- [How to Contribute](#how-to-contribute)


## Installation

Introduce how to install and deploy the code

### Requirements
Give a list of dependencies on packages

### Dataset


## User Guide

Introduce how to run the code from the command line, how to run the code from IDE, how to fine-tune the model, the structure of code, the function of different directories, how to load graphs, how to evaluate the models.

## Implemented Models

| Model  | Paper  | Venue  | Reference  |
|-------|--------|--------|--------|
| **GraphConsis** | Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection  | SIGIR 2020  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/graphconsis.txt) |
| **SemiGNN** | [A Semi-supervised Graph Attentive Network for Financial Fraud Detection](https://ieeexplore.ieee.org/abstract/document/8970829)  | ICDM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/semignn.txt) |
| **Player2Vec** | [Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding Framework](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf)  | CIKM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/player2vec.txt)|
| **GAS** | [Spam Review Detection with Graph Convolutional Networks](https://arxiv.org/abs/1908.10679)  | CIKM 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gas.txt) |
| **FdGars** | [FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System](https://dl.acm.org/citation.cfm?id=3316586)  | WWW 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/fdgars.txt) |
| **GeniePath** | [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910)  | AAAI 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/geniepath.txt)  |
| **GEM** | [Heterogeneous Graph Neural Networks for Malicious Account Detection](https://dl.acm.org/citation.cfm?id=3272010)  | CIKM 2018 |[BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gem.txt) |
<!--| **HACUD** | [Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism](https://aaai.org/ojs/index.php/AAAI/article/view/3884)  | AAAI 2019 |  Bibtex |-->

## Model Comparison
| Model  | Application  | Graph Type  | Base Model  |
|-------|--------|--------|--------|
| **GraphConsis** | Opinion Fraud  | Homogeneous   | GraphSAGE |
| **SemiGNN** | Financial Fraud  | Heterogeneous   | GAT, LINE, DeepWalk |
| **Player2Vec** | Cyber Criminal  | Heterogeneous | GAT, GCN|
| **GAS** | Opinion Fraud  | Heterogeneous | GCN, GAT |
| **FdGars** |  Opinion Fraud | Homogeneous | GCN |
| **GeniePath** | Financial Fraud | Homogeneous | GAT  |
| **GEM** | Financial Fraud  | Heterogeneous |GCN |
<!--| **HACUD** |  |  |   |-->

## TODO List
- The implementation of GraphConsis
- Add preprocessed Yelp datasets
- The memory-efficient implementation of SemiGNN
- The log loss for GEM model
- Add sampling methods
- Benchmarking SOTA models
- Scalable Implementation
- Pytorch Version

## How to Contribute
You are welcomed to contribute to this open-source toolbox. The detailed instructions will be released soon. Currently, you can create issues or send email to [ytongdou@gmail.com](mailto:ytongdou@gmail.com) for enquiry.

