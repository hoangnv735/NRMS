import random
import numpy as np
from typing import List
import orjson as json
import torch
from gensim.models import KeyedVectors
from torch.utils import data
from tqdm import tqdm


class Dataset(data.Dataset):
    def __init__(
        self, 
        article_file: str, 
        user_file: str, 
        w2v, 
        maxlen: int,
        npratio: int,
        his_len : int):
        self.articles = self.load_json(article_file)
        self.users = self.load_json(user_file)
        self.maxlen = maxlen
        self.npratio = npratio
        self.his_len = his_len

        if w2v != None:
            self.w2id = {w: w2v.vocab[w].index for w in w2v.vocab}
        else:
            self.w2id = torch.load('./data/articles_with_VnCoreNLP/w2id.pt')

    def load_json(self, file: str):
        with open(file, 'r') as f:
            return json.loads(f.read())

    def sent2idx(self, tokens: List[str]):
        # tokens = tokens[3:]
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        """getitem

        Args:
            idx (int): 
        Data:
            return (
                user_id (int): 1
                click (tensor): [batch, num_click_docs, seq_len]
                cand (tensor): [batch, num_candidate_docs, seq_len]
                label: candidate docs label (0 or 1)
            )
        """
        n_history = self.his_len
        history = self.users[idx]['history']
        push = self.users[idx]['push']
        if len(history) > n_history:
            random.shuffle(history)
        else:
            history = history * (n_history // len(history) + 1)
        uid = self.users[idx]['user_id']
        click_doc = [
            self.sent2idx(
                self.articles[p]['title']) for p in history[:n_history]
            ]
        cand_doc = []
        cand_doc_label = []

        #pos
        cand_doc.append(self.sent2idx(
            self.articles[push[random.randint(0, len(push) - 1)]]['title']))
        cand_doc_label.append(1)
        # neg
        for i in range(self.npratio):
            neg_id = -1
            while (neg_id == -1) \
                or (neg_id == 0) \
                or (neg_id in history) \
                or (neg_id in push):
                neg_id = random.randint(0, len(self.articles) - 1)
            cand_doc.append(self.sent2idx(self.articles[neg_id]['title']))
            cand_doc_label.append(0)

        # # neg
        # for i in range(self.neg_k):
        #     neg_id = -1
        #     while (neg_id == -1) \
        #         or (neg_id == 0) \
        #         or (neg_id in history) \
        #         or (neg_id in push):
        #         neg_id = random.randint(0, len(self.articles) - 1)
        #     cand_doc.append(self.sent2idx(self.articles[neg_id]['title']))
        #     cand_doc_label.append(0)
        # # pos
        # try:
        #     cand_doc.append(self.sent2idx(
        #         self.articles[push[
        #             random.randint(self.pos_num, len(self.push) - 1)
        #             ]['title']]))
        #     cand_doc_label.append(1)
        # except Exception:
        #     try:
        #         cand_doc.append(self.sent2idx(
        #             self.articles[push[0]]['title']))
        #     except:
        #         print(push[0])
        #         print(self.articles[push[0]])
        #     cand_doc_label.append(1)

        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        # return torch.tensor(click_doc), \
        #     torch.tensor(cand_doc), \
        #         torch.tensor(cand_doc_label, dtype=torch.float).argmax(0)
        return torch.LongTensor(click_doc), \
            torch.LongTensor(cand_doc), \
                torch.tensor(cand_doc_label, dtype=torch.float).argmax(0)
                # torch.LongTensor(cand_doc_label)#

class ValDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        articles_path, 
        users_list_path,
        all_items_path, 
        w2v, 
        maxlen,
        his_len,
        pos,
        neg,
        total):
        with open(articles_path, 'r') as f:
            self.articles = json.loads(f.read())
        with open(users_list_path, 'r') as f:
            self.users = json.loads(f.read())
        self.val_items = torch.load(all_items_path) 
        if w2v != None:
            self.w2id = {w: w2v.vocab[w].index for w in w2v.vocab}
        else:
            self.w2id = torch.load('./data/articles_with_VnCoreNLP/w2id.pt')
        self.maxlen = maxlen
        self.his_len = his_len
        self.pos = pos
        self.neg = neg
        self.total = total

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
        n_history = self.his_len
        uid = self.users[idx]['user_id']
        history = self.users[idx]['history']
        
        if len(history) > n_history:
            random.shuffle(history)
        else:
            history = history * (n_history // len(history) + 1)
        push = self.users[idx]['push']
        neg = [item 
            for item in self.val_items 
                if (item not in history and item not in push)]

        np.random.shuffle(push)
        np.random.shuffle(neg)
        np.random.shuffle(history)

        if (self.pos != 0 and self.neg != 0):            
            if len(push) >= self.pos:
                push = push[:self.pos]
                neg = neg[:self.neg]
            else:
                total = self.pos + self.neg
                neg = neg[:(total - len(push))]
        
        labels = np.array([1] * len(push) + [0] * len(neg))
        viewed = [
            self.sent2idx(
                self.articles[v]['title']
                ) for v in history[:n_history]
            ]
        cands = push + neg
        if self.pos == 0 and self.neg == 0:
            cands = cands[:self.total]
            labesl = labels[:self.total]
        cands = [self.sent2idx(self.articles[v]['title']) for v in cands]
        return torch.LongTensor(viewed), torch.LongTensor(cands),\
            torch.LongTensor(labels)


class TestDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        articles_path, 
        users_list_path,
        all_items_path, 
        w2v, 
        maxlen,
        his_len,
        pos,
        neg,
        total):
        with open(articles_path, 'r') as f:
            self.articles = json.loads(f.read())
        with open(users_list_path, 'r') as f:
            self.users = json.loads(f.read())
        self.test_items = torch.load(all_items_path) 
        if w2v != None:
            self.w2id = {w: w2v.vocab[w].index for w in w2v.vocab}
        else:
            self.w2id = torch.load('./data/articles_with_VnCoreNLP/w2id.pt')
        self.maxlen = maxlen
        self.his_len = his_len
        self.pos = pos
        self.neg = neg
        self.total = total

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
        n_history = self.his_len
        uid = self.users[idx]['user_id']
        history = self.users[idx]['history']
        if len(history) > n_history:
            random.shuffle(history)
        else:
            history = history * (n_history // len(history) + 1)
        push = self.users[idx]['push']
        neg = [item 
            for item in self.test_items 
                if (item not in history and item not in push)]

        np.random.shuffle(push)
        np.random.shuffle(neg)
        np.random.shuffle(history)

        if (self.pos != 0 and self.neg != 0):            
            if len(push) >= self.pos:
                push = push[:self.pos]
                neg = neg[:self.neg]
            else:
                total = self.pos + self.neg
                neg = neg[:(total - len(push))]
                
        labels = np.array([1] * len(push) + [0] * len(neg))
        viewed = [
            self.sent2idx(
                self.articles[v]['title']
                ) for v in history[:n_history]
            ]
        cands = push + neg
        if self.pos == 0 and self.neg == 0:
            cands = cands[:self.total]
            labels = labels[:self.total]
        cands = [self.sent2idx(self.articles[v]['title']) for v in cands]
        return torch.LongTensor(viewed), torch.LongTensor(cands),\
            torch.LongTensor(labels)

# class ValDataset(Dataset):
#     def __i# class ValDataset(Dataset):
#     def __init__(self, num, *args, **kwargs) -> None:
#         super(ValDataset, self).__init__(*args, **kwargs)
#         self.num = num
    
#     def __getitem__(self, idx: int):
#         n_history = self.his_len
#         push = self.users[idx]['push']
#         history = self.users[idx]['history']
#         random.shuffle(history)
#         random.shuffle(push)
#         uid = self.users[idx]['user_id']
#         click_doc = [
#             self.sent2idx(
#                 self.articles[p]['title']
#                 ) for p in history[:n_history]
#             ]
        
#         true_num = 5
#         #true_num = random.randint(1, min(self.num, len(push)) )
#         f_num = self.num - true_num
#         heso = true_num // len(push) + 1
#         cand_doc = random.sample(push*heso, true_num) # true
#         cand_doc_label = [1] * true_num
#         cand_doc.extend(random.sample(range(0, len(self.articles)), f_num)) # false
#         cand_doc_label.extend([0] * f_num)
#         tmp = list(zip(cand_doc, cand_doc_label))
#         random.shuffle(tmp)
#         cand_doc, cand_doc_label = zip(*tmp)
#         cand_doc = [
#             self.sent2idx(
#                 self.articles[cand]['title']
#                 ) for cand in cand_doc
#             ]
#         # print(torch.LongTensor(click_doc), torch.LongTensor(cand_doc), torch.LongTensor(cand_doc_label))
#         return torch.LongTensor(click_doc), \
#             torch.LongTensor(cand_doc), \
#                 torch.LongTensor(cand_doc_label)nit__(self, num, *args, **kwargs) -> None:
#         super(ValDataset, self).__init__(*args, **kwargs)
#         self.num = num
    
#     def __getitem__(self, idx: int):
#         n_history = self.his_len
#         push = self.users[idx]['push']
#         history = self.users[idx]['history']
#         random.shuffle(history)
#         random.shuffle(push)
#         uid = self.users[idx]['user_id']
#         click_doc = [
#             self.sent2idx(
#                 self.articles[p]['title']
#                 ) for p in history[:n_history]
#             ]
        
#         true_num = 5
#         #true_num = random.randint(1, min(self.num, len(push)) )
#         f_num = self.num - true_num
#         heso = true_num // len(push) + 1
#         cand_doc = random.sample(push*heso, true_num) # true
#         cand_doc_label = [1] * true_num
#         cand_doc.extend(random.sample(range(0, len(self.articles)), f_num)) # false
#         cand_doc_label.extend([0] * f_num)
#         tmp = list(zip(cand_doc, cand_doc_label))
#         random.shuffle(tmp)
#         cand_doc, cand_doc_label = zip(*tmp)
#         cand_doc = [
#             self.sent2idx(
#                 self.articles[cand]['title']
#                 ) for cand in cand_doc
#             ]
#         # print(torch.LongTensor(click_doc), torch.LongTensor(cand_doc), torch.LongTensor(cand_doc_label))
#         return torch.LongTensor(click_doc), \
#             torch.LongTensor(cand_doc), \
#                 torch.LongTensor(cand_doc_label)

# class TestDataset(data.Dataset):
#     def __init__(self, article_file: str, user_file: str, w2v, maxlen: int = 20, pos_num: int = 5, neg_k: int = 20):
#         self.articles = self.load_json(article_file)
#         self.users = self.load_json(user_file)
#         self.maxlen = maxlen
#         self.neg_k = neg_k
#         self.pos_num = pos_num

#         self.w2id = {w: w2v.vocab[w].index for w in w2v.vocab}

#     def load_json(self, file: str):
#         with open(file, 'r') as f:
#             return json.loads(f.read())

#     def sent2idx(self, tokens: List[str]):
#         # tokens = tokens[3:]
#         if ']' in tokens:
#             tokens = tokens[tokens.index(']'):]
#         tokens = [self.w2id[token.strip()]
#                   for token in tokens if token.strip() in self.w2id.keys()]
#         tokens += [0] * (self.maxlen - len(tokens))
#         tokens = tokens[:self.maxlen]
#         return tokens

#     def __len__(self):
#         return len(self.users)

#     def __getitem__(self, idx: int):
#         """getitem

#         Args:
#             idx (int): 
#         Data:
#             return (
#                 user_id (int): 1
#                 click (tensor): [batch, num_click_docs, seq_len]
#                 cand (tensor): [batch, num_candidate_docs, seq_len]
#                 label: candidate docs label (0 or 1)
#             )
#         """
#         n_history = 50
#         history = self.users[idx]['history']
#         push = self.users[idx]['push']
#         random.shuffle(history)
#         history = history[:n_history]
#         uid = self.users[idx]['user_id']
#         click_doc = [self.sent2idx(self.articles[p]['title']) for p in history]
#         cand_doc = []
#         cand_doc_label = []
#         # neg
#         for i in range(self.neg_k):
#             neg_id = -1
#             while (neg_id == -1) or (neg_id == 0) or (neg_id in history) or (neg_id in push):
#                 neg_id = random.randint(0, len(self.articles) - 1)
#             cand_doc.append(self.sent2idx(self.articles[neg_id]['title']))
#             cand_doc_label.append(0)
#         # pos
#         try:
#             cand_doc.append(self.sent2idx(
#                 self.articles[push[random.randint(pos_num, len(self.push) - 1)]['title']]))
#             cand_doc_label.append(1)
#         except Exception:
#             try:
#                 cand_doc.append(self.sent2idx(self.articles[push[0]]['title']))
#             except:
#                 print(push[0])
#                 print(self.articles[push[0]])
#             cand_doc_label.append(1)

#         tmp = list(zip(cand_doc, cand_doc_label))
#         random.shuffle(tmp)
#         cand_doc, cand_doc_label = zip(*tmp)
#         return torch.tensor(click_doc), torch.tensor(cand_doc), torch.tensor(cand_doc_label, dtype=torch.float).argmax(0)