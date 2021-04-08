# GraphSAGE

## Paper
The GraphSAGE model is proposed by the [paper](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) below:
```bibtex
@inproceedings{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  booktitle={Advances in neural information processing systems},
  pages={1024--1034},
  year={2017}
}
```

# Brief Introduction
We revise the code of [graphsage](https://github.com/subbyte/graphsage-tf2) so that it can load our data format and train the model.

# Run the code
`python GraphSage_main.py`

# Note
- Since graphsage only supports one type of relation, hence we only use the one major relation as the adjacency matrix. In the code, it only reads `rur` relation. You may change to all relations by change the parameter `meta=True` of `load_data_yelp` function in line 158 of `GraphSage_main.py` file
```python
load_data_yelp(meta=False, train_size=args.train_size)
```
- Before running the code, please remember unzip the given YelpChi dataset. 
