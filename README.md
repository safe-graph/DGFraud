<h3 align="center">
<p>Under Constuction Now. The first version is expected to be released in mid May, 2020.
</h3>

<p align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/194400/49531010-48dad180-f8b1-11e8-8d89-1e61320e1d82.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
</p>

<h3 align="center">
<p>A Deep Graph-based Tool Box for Fraud Detection
</h3>

Introduction: **DGFraud** is a Graph Neural Network (GNN) based toolbox for fraud detection. It integrates the implementation & comparison of state-of-the-art GNN-based fraud detection models. It also include several utility functions such as graph preprocessing, graph sampling, and performance evaluation. The introduction of implemented models can be found here. <!-- (Add introduction blogs links). -->

Contributed Users: Yutong Deng, BDSC Lab.

Welcome contribution, refer to the to-do list.

**Citation Information**

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
- [Contribute](#contribute)
- [License](#license)


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
| **SemiGNN** | [A Semi-supervised Graph Attentive Network for Financial Fraud Detection](https://github.com/yutongD/Player2Vec/tree/yingtong_modification/papers/SemiGNN.pdf)  | ICDM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/semignn.txt) |
| **Player2Vec** | [Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding Framework](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf)  | CIKM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/player2vec.txt)|
| **GAS** | [Spam Review Detection with Graph Convolutional Networks](https://arxiv.org/abs/1908.10679)  | CIKM 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gas.txt) |
| **FdGars** | [FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System](https://dl.acm.org/citation.cfm?id=3316586)  | WWW 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/fdgars.txt) |
| **GeniePath** | [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910)  | AAAI 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/geniepath.txt)  |
| **GEM** | [Heterogeneous Graph Neural Networks for Malicious Account Detection](https://dl.acm.org/citation.cfm?id=3272010)  | CIKM 2018 |[BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gem.txt) |
<!--| **HACUD** | [Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism](https://aaai.org/ojs/index.php/AAAI/article/view/3884)  | AAAI 2019 |  Bibtex |-->

## Model Comparison
| Model  | Application  | Graph Type  | Base Model  |
|-------|--------|--------|--------|
| **SemiGNN** | Financial Fraud  | Heterogeneous   | GAT, LINE, DeepWalk |
| **Player2Vec** | Cyber Criminal  | Heterogeneous | GAT, GCN|
| **GAS** | Spam Detection  | Heterogeneous | GCN, GAT |
| **FdGars** |  Spam Detection | Homogeneous | GCN |
| **GeniePath** | Financial Fraud | Homogeneous | GAT  |
| **GEM** | Financial Fraud  | Heterogeneous |GCN |
<!--| **HACUD** |  |  |   |-->

## TODO List
- The memory-efficient implementation of SemiGNN
- The log loss for GEM model
- Comparsion between different models
- Add sampling methods
- Benchmarking SOTA models
- Scalable Implementation
- Pytorch Version

## Contribute


## License


