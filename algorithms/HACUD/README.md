#  HACUD

## Paper

The HACUD model is proposed by the [paper](https://aaai.org/ojs/index.php/AAAI/article/view/3884) below:

```bibtex
@inproceedings{DBLP:conf/aaai/HuZSZLQ19,
  author    = {Binbin Hu and
               Zhiqiang Zhang and
               Chuan Shi and
               Jun Zhou and
               Xiaolong Li and
               Yuan Qi},
  title     = {Cash-Out User Detection Based on Attributed Heterogeneous Information
               Network with a Hierarchical Attention Mechanism},
  booktitle = {The Thirty-Third AAAI Conference on Artificial Intelligence},
  year      = {2019}
}
```

## Run the code

Go to `algorithms/HACUD/`，and run the following command in the terminal:

`python main.py --dataset dblp --gpu 0 --epoch 100 --embed_size 16 --batch_size 64 --lr 1e-4 ` 

## Meaning of the arguments

```
--lr: learning rate
--gpu: gpu id
--epoch: number of training epoches
--embed_size: size of the hidden representations of nodes
--batch_size: training batch size
```

There are also several optional arguments for this model, read parse.py for details.

## Note

- Before running the code, please remember unzip the given dataset