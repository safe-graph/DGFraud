# Description
We revise the original code of [graphsage](https://github.com/williamleif/GraphSAGE/tree/master/graphsage) so that it can load our data format and train the model.

# Run the code
`python -m graphsage.supervised_train --train_prefix ../../dataset --model graphsage_mean --sigmoid`

# Note
- Since graphsage only supports one type of relation, hence we only use the one major relation as the adjacency matrix. In the code, it only reads `rur` relation. You may change it by revise
line 28 in `utils.py` file
```python
rownetworks = [data['net_rur']]
```
- Before running the code, please remember unzip the given dataset. 
