import os
import sys

import torch
import random
import argparse
import numpy as np
from datetime import datetime


# get relataive path
from DPGraphGen.model import DPGraphGen
from GraphGen.model import GraphGen
from logger import stat_logger
from DPGraphGen import adadp
from config import load_config
from dataloader import Single_Graph_Dataset, Multi_Graph_Dataset
from train import train

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))
single_graph_dataset = set(['cora', 'karate', 'citeseer'])
multi_graph_dataset = set(['dblp2','IMDB_MULTI'])


def arg_parser():
	# init the common args, expect the model specific args
	parser = argparse.ArgumentParser()
	parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
	parser.add_argument('--seed', type=int, default=43, help='Random seed.')
	parser.add_argument('--threads', type=int, default=4, help='Thread number.')
	parser.add_argument('--txt_log', type=bool, default=True, help='whether save the txt_log.')
	parser.add_argument('--model_name', type=str, default='DPGraphGen', help='[GraphGen, DPGraphGen]')
	parser.add_argument('--dataset_str', type=str, default='dblp2', help="['cora', 'karate', 'citeseer', 'dblp2', 'IMDB_MULTI']")
	parser.add_argument('--n_eigenvector', type=int, default=64, help='use eigenvector as feature.')

	parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
	parser.add_argument('--test_period', type=int, default=10, help='test period.')
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--learning_rate', default=0.02, help='the ratio of training set in whole dataset.')
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--stat_generate_num', type=int, default=5)
	parser.add_argument('--threshold', type=float, default=None)

	parser.add_argument('--eval_period', type=int, default=10)
	parser.add_argument('--draw_graph_epoch', type=int, default=2)
	parser.add_argument('--generated_graph_num', type=int, default=5, help='generated_graph_num is for presentation')

	parser.add_argument('--discriminator_ratio', type=float, default=0.5, help='factor of discriminator loss.')
	args = parser.parse_args()
	return args



def set_env(args):
	# get configuration for different model
	# set random seed
	# get current time for log
	# args.cuda = not args.no_cuda and torch.cuda.is_available()
	current_time = datetime.now().strftime('%b_%d_%H-%M-%S')
	model_args = load_config(args.model_name, args.dataset_str, args.optimizer)
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.set_num_threads(args.threads)

	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
	return args, model_args, current_time


def load_data(args, dataset_str):
	n_eigenvector = args.n_eigenvector
	if dataset_str in single_graph_dataset:
		args.dataset_type = 'single'
		dataset = Single_Graph_Dataset(dataset_str, n_eigenvector)
		args.num_samples = dataset.features.shape[0]
	elif dataset_str in multi_graph_dataset:
		args.dataset_type = 'multi'
		dataset = Multi_Graph_Dataset(dataset_str, n_eigenvector)
		args.num_samples = dataset.datasets[0].features.shape[0]
	else:
		print('unkown dataset name, load_data.')
		sys.exit(1)

	return dataset


def create_model(args, model_args, dataset):
	'''
	initialize model

	features: train graph node feature
	adj_lists_train: train graph adj
	'''

	if args.batch_size > dataset.features.shape[0]:
		args.batch_size = dataset.features.shape[0]
	model = None
	if args.model_name == 'GraphGen':
		model = GraphGen(args, model_args, dataset.features, dataset.adj_list)

	elif args.model_name == 'DPGraphGen':
		model = DPGraphGen(args, model_args, dataset.features, dataset.adj_list)

		is_enc1 = True
		for v in model.parameters():
			if v is not None and v.requires_grad is True:
				if len(v.data.shape) == 2:
					continue
				if is_enc1:
					is_enc1 = False
					v.data.copy_(v[0].data.clone().repeat(args.batch_size * (1 + model_args.samp_num), 1, 1))
				else:
					v.data.copy_(v[0].data.clone().repeat(args.batch_size, 1, 1))
	return model



def create_optimizer(args, model):
	optimizer = None
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
	elif args.optimizer == 'SGD':
		optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
	elif args.optimizer == 'ADADP':
		optimizer = adadp.ADADP(filter(lambda p: p.requires_grad, model.parameters()))
	return optimizer


def main(args, model_args, dataset):
	model = create_model(args, model_args, dataset)
	optimizer = create_optimizer(args, model)
	logger = stat_logger(args, current_time)
	train(args, model_args, dataset, model, optimizer, logger)


if __name__ == '__main__':
	args = arg_parser()

	args, model_args, current_time = set_env(args)
	datasets = load_data(args, args.dataset_str)
	if args.dataset_type == 'single':
		model_args.dec2_dim = datasets.actual_feature_dim
		main(args, model_args, datasets)
	else:
		for counter, dataset in enumerate(datasets.datasets):
			model_args.dec2_dim = dataset.actual_feature_dim
			print('='*10 + str(counter) + '='*10)
			main(args, model_args, dataset)