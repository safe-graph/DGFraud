# Description
This is the code for our graphconsis mode. It is revised based on the [graphsage](https://github.com/williamleif/GraphSAGE/tree/master/graphsage) model. We support multiple relations and distance sampling as mentioned in [our paper](https://arxiv.org/pdf/2005.00625.pdf).

# Run the code
`python -m hetegraphconsis.supervised_train --train_prefix ../../dataset --file_name YelpChi.mat --model graphsage_mean --sigmoid True --epochs 3 --samples_1 10 -samples_2 5 --context_dim 128 --gpu 1`

## Meaning of the arguments
```
--samples_1 -samples_2: the number of samples using at difference layers
--context_dim: the dimension of context embeddings
```
For more information about the arguments, please refer to `supervised_train.py`.
# Note
- the major differences of graphsage and graphconsis are in the `neighbor_sampler.py' and 'supervised_models.py'. For neighbor_sampler, we use distance sampler that computing the consistency score and sampling probability. For supervised model, the graphconsis model considers all the relations and learn relation vectors and attention weights for each sample.

- Before running the code, please remember unzip the given dataset. 
