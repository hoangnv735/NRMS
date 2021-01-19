import os
import torch
from argparse import ArgumentParser
from trainer import Trainer
from config import hparams


parser = ArgumentParser('train args')
parser.add_argument('--des', default=None, type=str, help='description')
parser.add_argument('--epochs', default=None, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=None, type=int, help='batch_size')
parser.add_argument('--start_epoch', default=None, type=int, help='epoch count from')
parser.add_argument('--val_pos', default=None, type=int, help='val_pos')
parser.add_argument('--val_neg', default=None, type=int, help='val_neg')
parser.add_argument('--val_total', default=None, type=int, help='val_total')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--embedding', default=None, type=str, help='pretrain embedding')
parser.add_argument('--train_data', default=None, type=str, help='train data')
parser.add_argument('--all_items', default=None, type=str, help='all items')
parser.add_argument('--val_data', default=None, type=str, help='val data')
parser.add_argument('--val_items', default=None, type=str, help='val items')
parser.add_argument('--load_model', default=None, type=str, help='path to pretrain model')
# parser.add_argument('--save_path', default=None, type=str, help='path to save model')

args = parser.parse_args()

if args.epochs == None: raise ValueError('missing epochs')     
if args.start_epoch == None: raise ValueError('nissing description')   
if args.batch_size == None: raise ValueError('missing batch_size')        
if args.embedding == None: print('Train with no embedding')
if args.train_data == None: raise ValueError('missing train_data')
if args.all_items == None: raise ValueError('missing all_items')
if args.val_data == None: raise ValueError('missing val_data')
if args.val_items == None: raise ValueError('missing val_items')
if args.val_pos == None: raise ValueError('missing val_items')
if args.val_neg == None: raise ValueError('missing val_pos')
if args.val_total == None: raise ValueError('missing val_neg')
if args.lr == None: raise ValueError('missing learning rate')

# if args.save_path == None: raise ValueError('missing save_path')


hparams['epochs'] = args.epochs
hparams['batch_size'] = args.batch_size
hparams['embedding'] = args.embedding
hparams['train_data'] = args.train_data
hparams['all_items'] = args.all_items
hparams['val_data'] = args.val_data
hparams['val_items'] = args.val_items
hparams['val']['pos'] = args.val_pos
hparams['val']['neg'] = args.val_neg
hparams['val']['total'] = args.val_total
hparams['lr'] = args.lr
hparams['start_epoch'] = args.start_epoch
hparams['des'] = args.des
# hparams['save_path'] = args.save_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(hparams, device)
if args.load_model != None:
  print('Found pretrained. Loading model...',end=' ')
  trainer.load_model(args.load_model)
  print('Done')
trainer.fit()