import pytorch_lightning as pl
import torch
import os
import numpy as np
import orjson as json
from tqdm import tqdm
from model import NRMS
from gensim.models import Word2Vec, KeyedVectors
from typing import List
from pytorch_lightning import Trainer
from argparse import ArgumentParser


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, articles_path, users_list_path, all_items_path, w2v, maxlen=20):
        with open(articles_path, 'r') as f:
            self.articles = json.loads(f.read())
        with open(users_list_path, 'r') as f:
            self.users = json.loads(f.read())
        self.test_items = torch.load(all_items_path) 
        self.w2id = {w: w2v.vocab[w].index for w in w2v.vocab}
        self.maxlen = maxlen

    def __len__(self):
        return len(self.users)
        # return 10

    def sent2idx(self, tokens: List[str]):
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens

    def __getitem__(self, idx):
        uid = self.users[idx]['user_id']
        history = self.users[idx]['history']
        push = self.users[idx]['push']
        neg = [item for item in self.test_items if item not in history]

        np.random.shuffle(push)
        np.random.shuffle(neg)
        np.random.shuffle(history)

        push = push[:5]
        neg = neg[:20]
        
        labels = np.array([1] * len(push) + [0] * len(neg))
        viewed = [self.sent2idx(self.articles[v]['title']) for v in history[:50]]
        cands = push + neg
        # print("push: ", len(push))
        # print("len neg: ", len(neg))
        # cands = cands[:400]
        # labels = labels[:400]
        cands = [self.sent2idx(self.articles[v]['title']) for v in cands]
        return torch.LongTensor(viewed), torch.LongTensor(cands),\
            torch.LongTensor(labels)

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.w2v = KeyedVectors.load_word2vec_format(
            hparams['pretrained_model'], 
            binary=True)

        if hparams['model']['dct_size'] == 'auto':
            hparams['model']['dct_size'] = len(self.w2v.vocab)

        self.model = NRMS(hparams['model'], torch.tensor(self.w2v.vectors))
        self.hparams = hparams
        self.maxlen = hparams['data']['maxlen']
        # self.test_ds = TestDataset(
        #     articles_path,
        #     users_list_path,
        #     all_items_path,
        #     self.w2v)
        with open(articles_path, 'r') as f:
            self.articles = json.loads(f.read())
        with open(users_list_path, 'r') as f:
            self.users = json.loads(f.read())
        self.test_items = torch.load(all_items_path) 
        self.w2id = {w: self.w2v.vocab[w].index for w in self.w2v.vocab}
        self.maxlen = 20

    def sent2idx(self, tokens: List[str]):
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens

    def predict_one(self):
        idx = np.random.randint(0,len(self.users)-1)
        push = self.users[idx]['push']
        history = self.users[idx]['history']
        neg = [item for item in self.test_items if item not in history]

        np.random.shuffle(push)
        np.random.shuffle(neg)
        np.random.shuffle(history)

        push = push[:5]
        neg = neg[:20]

        labels = np.array([1] * len(push) + [0] * len(neg))
        viewed = [self.articles[v]['title'] for v in history[:50]]
        cands = push + neg
        cands = [self.articles[v]['title'] for v in cands]
        print("Users history:")
        self.print_string(viewed)
        print("Candidates:")
        print("\nActual click")
        self.print_string([self.articles[v]['title'] for v in push])
        print("\nNo click")
        self.print_string([self.articles[v]['title'] for v in neg])


        # cands = cands[:400]
        # labels = labels[:400]
        viewed = [self.sent2idx(v) for v in viewed]
        o_cands = list(cands)
        cands = [self.sent2idx(c) for c in cands]
        viewed = torch.LongTensor([viewed])
        cands = torch.LongTensor([cands])
        labels = torch.LongTensor(labels)        
        with torch.no_grad():
            logits = self.model(viewed, cands)
        val, idx = logits.topk(10)
        val = val.squeeze().detach().cpu().numpy()
        results = [o_cands[int(i)] 
          for i in idx.squeeze().detach().cpu().numpy()
          ]
        print("\nPredict results")
        for score, title in zip(val, results):
            out = " ".join(title)
            print('{}  {}'.format(round(score,4), out))

    def print_string(self, str_list):
        for string in str_list:
            out = " ".join(string)
            print(out)
            

    # = './data/data/articles.json'
    # users_list_path = './data/data/data_update/test/users_list_21.json'
    # all_items_path = './data/data/data_training/negative/day21.pt'

    # pretrain_path = './lightning_logs/checkpoint/15_epoch_train_day21.ckpt'

    # batch_size = 1
if __name__ == '__main__':
    parser = ArgumentParser('train args')
    parser.add_argument('--batch_size', default=None, type=int, help='batch_size')
    parser.add_argument('--embedding', default=None, type=str, help='pretrain embedding')
    parser.add_argument('--all_items', default=None, type=str, help='all items')
    parser.add_argument('--test_data', default=None, type=str, help='test data')
    parser.add_argument('--test_items', default=None, type=str, help='test items')
    parser.add_argument('--load_model', default=None, type=str, help='path to pretrain model')

    args = parser.parse_args()
    if args.all_items == None: raise ValueError('missing all_items')
    if args.test_data == None: raise ValueError('missing test_data')
    if args.test_items == None: raise ValueError('missing test_items')
    if args.load_model == None: raise ValueError('missing load_model')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = args.batch_size
    articles_path = args.all_items
    users_list_path = args.test_data
    all_items_path = args.test_items
    pretrain_path = args.load_model
    # articles_path = './data/data/articles.json'
    # users_list_path = './data/data/data_update/test/users_list_21.json'
    # all_items_path = './data/data/data_training/negative/day21.pt'

    # pretrain_path = './lightning_logs/checkpoint/15_epoch_train_day21.ckpt'

    # batch_size = 1
    nrms = Model.load_from_checkpoint(pretrain_path)
    nrms.predict_one()