import torch
import time
import numpy as np
from torch.utils import data
from gensim.models import KeyedVectors
from model.net import NRMS
from argparse import ArgumentParser
from config import hparams
from dataset import Dataset, ValDataset, TestDataset

# def dcg_score_old(y_true, y_score, k=10):
#     k = min(k, len(y_true))
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#     gains = 2 ** y_true - 1
#     discounts = np.log2(np.arange(len(y_true)) + 2)
#     return np.sum(gains / discounts)

# def ndcg_score_old(y_true, y_score, k=10):
#     k = min(k, len(y_true))
#     best = dcg_score_old(y_true, y_true, k)
#     actual = dcg_score_old(y_true, y_score, k)
#     return actual / best

# def mrr_score_old(y_true, y_score):
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order)
#     rr_score = y_true / (np.arange(len(y_true)) + 1)
#     return np.sum(rr_score) /  np.sum(y_true)

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def hit_score(y_true, y_score, k=10):
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0

def dcg_score(y_true, y_score, k=10):
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

class Trainer(object):
    def __init__(self, hparams, device):
        self.device = device
        if hparams['embedding'] != None:
            self.w2v = KeyedVectors.load_word2vec_format(
                hparams['embedding'], 
                binary=True)
            self.vector = torch.tensor(self.w2v.vectors)
            if hparams['model']['dct_size'] == 'auto':
                hparams['model']['dct_size'] = len(self.w2v.vocab)
        else:
            self.w2v = None
            self.vector = None
            
            # if hparams['model']['dct_size'] == 'auto':
            #     hparams['model']['dct_size'] = 22537
            
        self.model = NRMS(hparams['model'], self.vector).to(self.device)
        self.hparams = hparams
        self.maxlen = hparams['title_len']
        self.optm = self.configure_optimizers()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams['lr'], 
            weight_decay=1e-5)

    def prepare_data(self):
        """prepare_data

        load dataset
        """       
        print('Prepating train data...', end=' ')
        self.train_ds = Dataset(
            self.hparams['all_items'], 
            self.hparams['train_data'],
            self.w2v, 
            maxlen=self.hparams['title_len'],  
            npratio=self.hparams['train']['npratio'],
            his_len = self.hparams['his_len'])
        
        self.val_ds = ValDataset(
            self.hparams['all_items'], 
            self.hparams['val_data'],
            self.hparams['val_items'], 
            self.w2v,
            maxlen=self.hparams['title_len'], 
            his_len=self.hparams['his_len'],
            pos=self.hparams['val']['pos'],
            neg=self.hparams['val']['neg'],
            total=self.hparams['val']['total'])
        print('Done')
        # tmp = [t.unsqueeze(0) for t in self.train_ds[0]]
        # self.logger.experiment.add_graph(self.model, tmp)
        # num_train = int(len(ds) * 0.85)
        # num_val = len(ds) - num_train
        # self.train_ds, self.val_ds = data.random_split(ds, (num_train, num_val))

    def prepare_test_data(self):
        print('Preparing test data...', end=' ')
        self.test_ds = TestDataset(
            articles_path=self.hparams['all_items'],
            users_list_path=self.hparams['test_data'],
            all_items_path=self.hparams['test_items'],
            w2v=self.w2v,
            maxlen=self.hparams['title_len'],
            his_len=self.hparams['his_len'],
            pos=self.hparams['test']['pos'],
            neg=self.hparams['test']['neg'],
            total=self.hparams['test']['total'])
        print('Done')

    def train_dataloader(self):
        """

        return:
            train_dataloader
        """
        return data.DataLoader(
            self.train_ds, 
            batch_size=self.hparams['batch_size'], 
            num_workers=1, 
            shuffle=True)

    def val_dataloader(self):
        """

        return:
            val_dataloader
        """
        # sampler = data.RandomSampler(
        #     self.val_ds, num_samples=10000, replacement=True)
        return data.DataLoader(
            self.val_ds,
            batch_size=self.hparams['batch_size'], 
            num_workers=1)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, 
            batch_size=self.hparams['test_batch_size'])

    # def training_step(self, batch, batch_idx):
    def training_step(self, batch):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        clicks, cands, labels = batch
        clicks = clicks.to(self.device)
        cands = cands.to(self.device)
        labels = labels.to(self.device)
        loss = self.model(clicks, cands, labels)
        loss.backward()
        self.optm.step()
        self.optm.zero_grad()
        return {'loss': loss.detach()}

    def training_epoch_end(self, outputs):
        """for each epoch end

        Arguments:
            outputs: list of training_step output
        """
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        return loss_mean
        # logs = {'train_loss': loss_mean}
        # self.model.eval()
        # self.logger.log_metrics(logs, self.current_epoch)
        #return {'progress_bar': logs, 'log': logs}

    # def validation_step(self, batch, batch_idx):
    def validation_step(self, batch):        
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        clicks, cands, cands_label = batch
        clicks = clicks.to(self.device)
        cands = cands.to(self.device)
        with torch.no_grad():
          logits = self.model(clicks, cands)
        mrr = 0.
        # auc = 0.
        ndcg5, ndcg10 = 0., 0.

        for score, label in zip(logits, cands_label):
            # auc += pl.metrics.functional.auroc(score, label)
            # score = score.view(-1).detach().cpu().numpy()
            # label = label.view(-1).detach().cpu().numpy()
            score = score.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            mrr += mrr_score(label, score)
            ndcg5 += ndcg_score(label, score, 5)
            ndcg10 += ndcg_score(label, score, 10)
            # order = np.argsort(score)
            # mrr += mrr_score(label, score, 5, order)
            # ndcg5 += ndcg_score(label, score, 5, order)
            # ndcg10 += ndcg_score(label, score, 10, order)

        return {
            # 'auroc': (auc / logits.shape[0]).item(), 
            'mrr': (mrr / logits.shape[0]).item(), 
            'ndcg5': (ndcg5 / logits.shape[0]).item(), 
            'ndcg10': (ndcg10 / logits.shape[0]).item() 
            }
        # return {
            # 'auroc': (auc / logits.shape[0]).item(), 
            # 'mrr': (mrr / logits.shape[0]), 
            # 'ndcg5': (ndcg5 / logits.shape[0]), 
            # 'ndcg10': (ndcg10 / logits.shape[0])
            # }

    def validation_epoch_end(self, outputs):
        """
        validation end

        Arguments:
            outputs: list of training_step output
        """
        mrr = torch.tensor([x['mrr'] for x in outputs])
        # auroc = torch.tensor([x['auroc'] for x in outputs])
        ndcg5 = torch.tensor([x['ndcg5'] for x in outputs])
        ndcg10 = torch.tensor([x['ndcg10'] for x in outputs])

        results = {
            # 'auroc': auroc.mean().item(), 
            'mrr': mrr.mean().item(), 
            'ndcg@5': ndcg5.mean().item(), 
            'ndcg@10': ndcg10.mean().item()
            }
        return results
        # self.logger.log_metrics(logs, self.current_epoch)
        # self.model.train()
        # return {'progress_bar': logs, 'log': logs}
        
    
    # def test_step(self, batch, batch_idx):
    def test_step(self, batch):

        viewed, cands, labels = batch
        viewed = viewed.to(self.device)
        cands = cands.to(self.device)
        with torch.no_grad():
            logits = self.model(viewed, cands)
        
        mrr = 0.
        auc = 0.
        ndcg1, ndcg5, ndcg10 = 0., 0., 0.
        # ndcg1, ndcg5, ndcg10, ndcg20 = 0., 0.,0., 0.
        hr1, hr5, hr10 = 0., 0., 0.

        for score, label in zip(logits, labels):
            # auc += pl.metrics.functional.auroc(score, label)
            score = score.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            mrr += mrr_score(label, score)

            ndcg1 += ndcg_score(label, score, 1)
            ndcg5 += ndcg_score(label, score, 5)
            ndcg10 += ndcg_score(label, score, 10)
            # ndcg20 += ndcg_score(label, score, 20)

            hr1 += hit_score(label, score, 1)
            hr5 += hit_score(label, score, 5)
            hr10 += hit_score(label, score, 10)

        return {
            # 'auroc': (auc / logits.shape[0]).item(), 
            'mrr': (mrr / logits.shape[0]).item(), 
            'ndcg1':(ndcg1 / logits.shape[0]).item(),
            'ndcg5': (ndcg5 / logits.shape[0]).item(), 
            'ndcg10': (ndcg10 / logits.shape[0]).item(),
            # 'ndcg20': (ndcg20 / logits.shape[0]).item(),
            'hr1':(hr1 / logits.shape[0]),
            'hr5': (hr5 / logits.shape[0]), 
            'hr10': (hr10 / logits.shape[0])
            }

    def test_epoch_end(self, outputs):
        mrr = torch.tensor([x['mrr'] for x in outputs])
        # auroc = torch.tensor([x['auroc'] for x in outputs])

        ndcg1 = torch.tensor([x['ndcg1'] for x in outputs])
        ndcg5 = torch.tensor([x['ndcg5'] for x in outputs])
        ndcg10 = torch.tensor([x['ndcg10'] for x in outputs])
        # ndcg20 = torch.tensor([x['ndcg20'] for x in outputs])

        hr1 = torch.tensor([x['hr1'] for x in outputs])
        hr5 = torch.tensor([x['hr5'] for x in outputs])
        hr10 = torch.tensor([x['hr10'] for x in outputs])

        results = {
            # 'auroc': auroc.mean(), 
            'mrr': mrr.mean(), 
            'ndcg@1': ndcg1.mean(), 
            'ndcg@5': ndcg5.mean(), 
            'ndcg@10': ndcg10.mean(),
            # 'ndcg@20': ndcg20.mean(),
            'hr@1': hr1.mean(), 
            'hr@5': hr5.mean(), 
            'hr@10': hr10.mean()
            }
        # self.logger.log_metrics(logs, self.current_epoch)
        # return {'progress_bar': logs, 'log': logs}
        return results

    def fit(self):
        print('Training on', self.device)
        self.prepare_data()
        train_dl = self.train_dataloader()
        val_dl = self.val_dataloader()
        first_epoch = self.hparams['start_epoch']
        epochs = self.hparams['epochs']
        last_epoch = epochs + first_epoch
        start_epoch = time.time()
        best_ndcg = 0.
        for epoch in range(first_epoch, last_epoch):
            start_time = time.time()
            print(f'Epoch {epoch}/{last_epoch-1}:', end=' ')
            outputs = []
            self.model.train()
            for batch_id, batch in enumerate(train_dl):
                # print(f'Training on batch {batch_id+1}/{len(train_dl)}')
                output = self.training_step(batch)
                outputs.append(output)
                # gc.collect()
            loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
            print('Loss: {:0.4f}'.format(loss_mean.item()), end=' ')
            end_time = time.time()
            print('Train epoch time: {:0.2f}s'.format(end_time-start_time))
            print()
            print('Validating')
            val_start = time.time()
            outputs = []
            self.model.eval()
            for batch in val_dl:
                output = self.validation_step(batch)
                outputs.append(output)
            results = self.validation_epoch_end(outputs)
            val_end = time.time()
            for k in results:
                print('{} {:0.4f}'.format(k, results[k]))
            print("Validation time: {:0.2f}s".format(val_end-val_start))

            # if results['ndcg@10'] > best_ndcg:
            #     best_ndcg = results['ndcg@10']
            #     print('Found better model, saving model')
            #     self.save_model(
            #         f"./checkpoint/{self.hpamras['des']}epoch={epoch}ndcg10={round(best_ndcg,4)}.pt"
            #         )
            
            ndcg10 = results['ndcg@10']
            self.save_model(
                f"./checkpoint/des={self.hparams['des']}epoch={epoch}ndcg10={round(ndcg10,4)}.pt"
                )
            print()    
        end_epoch = time.time()
        print("Train time: {:0.2f}s".format(end_epoch-start_epoch))
        print('\n')

    def test(self):
        print('Testing on', self.device)
        self.prepare_test_data()
        test_dl = self.test_dataloader()
        print("Testing...")
        t1 = time.time()
        outputs = []
        self.model.eval()
        for batch in test_dl:
            output = self.test_step(batch)
            outputs.append(output)
        results = self.test_epoch_end(outputs)
        for k in results:
            print('{} {:0.4f}'.format(k, results[k]))
        t2 = time.time()
        print("Test time: {:0.2f}s".format(t2-t1))

    def save_model(self, path):
        torch.save(
            self.model.state_dict(),
            path
        )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    
