# DGFraud
A graph neural network tool box for fraud detection

## Dataset
Preprocessed DBLP can be found in:<br/>
链接：https://pan.baidu.com/s/1L8GESaLKvVbM84ffp7h_mA 
提取码：cz0y <br/>
(copied data from Jhy1993/HAN)<br/> download and put in the /data directory

## Task Log
| Date   | Task  |  Assigned to  | Finished |
|-------|--------|--------|-------|
| 03/04 | Add pycharm helper files and .pyc files to .gitignore and delete them from repo | Yutong |<ul><li>- [x] </li></ul> | 
| 03/04 | Change the Player2Vec with multiple meta-graphs as input | Yutong |<ul><li>- [x] </li></ul> | 
| 03/04 | Change the hard-coded sparse dropout helper variable | Yutong |<ul><li>- [x] </li></ul> | 
| 02/18 | Run Player2Vec, FdGars and SpamGCN on Yelp spam review data | Yingtong |<ul><li>- [x] </li></ul> | 
| 02/18 | Implement SemiGNN | Yutong |<ul><li>- [ ] </li></ul> | 
| 02/18 | Add comments to all functions | Yutong |<ul><li>- [x] </li></ul> | 
| 02/18 | Solve the dblp dataset loading issue | Yutong |<ul><li>- [x] </li></ul> | 
| 12/02 | main.py: code structure refer to /reference/main.py | Yutong |<ul><li>- [x] </li></ul> | 
| 12/02 | main.py: move data reading functions to data_loader.py in /utils| Yutong |<ul><li>- [x] </li></ul> | 
| 12/02 | main.py: move nor_adj function to utils.py in /utils | Yutong |<ul><li>- [x] </li></ul> |
| 12/02 | Improve the header comments format: refer to /reference/gaussian_moments.py and /reference/px_expander.py | Yutong |<ul><li>- [x] </li></ul> | 
| 11/25 | Upload required files to make the code run regularly | Yutong |<ul><li>- [x] </li></ul> | 
| 11/25 | Put the Player2Vec class to a new file named player2vec.py, add header to explain the algorithm logic  | Yutong | <ul><li>- [x] </li></ul> |
| 11/25 | For each file in /models folder, add comment header to show the copyright and briefly introduce your change if you changed it  | Yutong | <ul><li>- [x] </li></ul> |
| 11/25 | Modify the FdGars   | Yutong | <ul><li>- [x] </li></ul> |
| 11/25 | Implement SpamGCN   | Yutong | <ul><li>- [x] </li></ul> |

## Algorithm Implementation Checklist
| Alg Name   | Title  | Venue |  Paper | Code  |
|-------|--------|--------|--------|-----------|
| **SemiGNN** | **A Semi-supervised Graph Attentive Network for Fraud Detection**  | ICDM 2019  |  [Link](https://github.com/yutongD/Player2Vec/tree/yingtong_modification/papers/SemiGNN.pdf)   |  <ul><li>- [ ] </li></ul> |
| **Player2Vec** | **Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding Framework**  | CIKM 2019  | [Link](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf) | <ul><li>- [x] </li></ul> |
| **GAS** | **Spam Review Detection with Graph Convolutional Networks**  | CIKM 2019  | [Link](https://arxiv.org/abs/1908.10679) | <ul><li>- [x] </li></ul> |
| **FdGars** | **FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System**  | The WebConference 2019 | [Link](https://dl.acm.org/citation.cfm?id=3316586) | <ul><li>- [x] </li></ul> |
| **HACUD** | **Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism**  | AAAI 2019 | [Link](https://aaai.org/ojs/index.php/AAAI/article/view/3884) | <ul><li>- [ ] </li></ul> |
| **GEM** | **Heterogeneous Graph Neural Networks for Malicious Account Detection**  | CIKM 2018 | [Link](https://dl.acm.org/citation.cfm?id=3272010) | <ul><li>- [ ] </li></ul> |
<!-- | 2019 | **Uncovering Insurance Fraud Conspiracy with Network Learning**  | SIGIR 2019 | [Link](https://dl.acm.org/citation.cfm?id=3331184.3331372) | Link | -->
<!-- | 2018 | **GraphRAD: A Graph-based Risky Account Detection System**  | MLG 2018 | [Link](https://www.mlgworkshop.org/2018/papers/MLG2018_paper_12.pdf) | Link | -->
