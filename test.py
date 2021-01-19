import torch
import os
import numpy as np
import orjson as json
from tqdm import tqdm
from model import NRMS
from gensim.models import Word2Vec, KeyedVectors
from config import hparams
from argparse import ArgumentParser
from trainer import Trainer


parser = ArgumentParser('test args')
parser.add_argument('--test_batch_size', default=None, type=int, help='test_batch_size')
parser.add_argument('--test_pos', default=None, type=int, help='test_pos')
parser.add_argument('--test_neg', default=None, type=int, help='test_neg')
parser.add_argument('--test_total', default=None, type=int, help='test_total')
parser.add_argument('--embedding', default=None, type=str, help='pretrain embedding')
parser.add_argument('--all_items', default=None, type=str, help='all items')
parser.add_argument('--test_data', default=None, type=str, help='test data')
parser.add_argument('--test_items', default=None, type=str, help='test items')
parser.add_argument('--load_model', default=None, type=str, help='path to pretrain model')


args = parser.parse_args()
if args.test_batch_size == None: raise ValueError('missing test_batch_size')
if args.test_pos == None: raise ValueError('missing test_pos')
if args.test_neg == None: raise ValueError('missing test_neg')
if args.test_total == None: raise ValueError('missing test_total')
if args.embedding == None: print('train without embedding')
if args.all_items == None: raise ValueError('missing all_items')
if args.test_data == None: raise ValueError('missing test_data')
if args.test_items == None: raise ValueError('missing test_items')
if args.load_model == None: raise ValueError('missing load_model')


hparams['test_batch_size'] = args.test_batch_size
hparams['test']['pos'] = args.test_pos
hparams['test']['neg'] = args.test_neg
hparams['test']['total'] = args.test_total
hparams['embedding'] = args.embedding
hparams['all_items'] = args.all_items
hparams['test_data'] = args.test_data
hparams['test_items'] = args.test_items
hparams['load_model'] = args.load_model
# articles_path = './data/data/articles.json'
# users_list_path = './data/data/data_update/test/users_list_21.json'
# all_items_path = './data/data/data_training/negative/day21.pt'
# pretrain_path = './lightning_logs/checkpoint/15_epoch_train_day21.ckpt'
# batch_size = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(hparams, device)
print('Loading model...',end=' ')
trainer.load_model(hparams['load_model'])
print('Done')
trainer.test()

