import sys
import torch
from torch._C import device
from torch.optim import SGD, Adam
from tqdm.auto import tqdm

from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F
from estimator import EstimateAdj, PGD, prox_operators
import time

class Trainer:
    def __init__(
            self,
            optimizer:      dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
            max_epochs:     dict(help='maximum number of training epochs') = 500,
            learning_rate:  dict(help='learning rate') = 0.01,
            weight_decay:   dict(help='weight decay (L2 penalty)') = 0.0,
            patience:       dict(help='early-stopping patience window size') = 0,
            orphic:         dict(help='use l1 norm', option='-pro') = False,
            inner_epoch:    dict(help='change estimator training ratio', option='-inner') = 2,
            device='cuda',
            logger=None,
            alpha = 5e-4,
            gamma = 1,
        ):
        self.optimizer_name = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.pro = orphic
        self.inner_epoch = inner_epoch
        self.logger = logger
        self.model = None
        self.alpha = alpha
        self.gamma = gamma
        self.estimator = None
        

    def configure_optimizers(self):
        if self.optimizer_name == 'sgd':
            return SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, model, data):
        self.model = model.to(self.device)
        data = data.to(self.device)

        optimizer = self.configure_optimizers()
        num_epochs_without_improvement = 0
        best_metrics = None

        epoch_progbar = tqdm(range(1, self.max_epochs + 1), desc='Epoch: ', leave=False, position=1, file=sys.stdout)
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            train_metrics = self._train(data, optimizer, epoch)
            metrics.update(train_metrics)

            val_metrics = self._validation(data)
            metrics.update(val_metrics)
            ''' start using pro-gnn loss function'''

            if self.pro:
                for i in range(self.inner_epoch):
                    perturbed_adj = to_dense_adj(data.edge_index)[0]
                    estimator = EstimateAdj(perturbed_adj, symmetric=False, device=self.device).to(self.device)
                    self.estimator = estimator
                    self.optimizer_adj = SGD(estimator.parameters(), momentum=0.9, lr=self.learning_rate)
                    self.optimizer_l1 = PGD(estimator.parameters(), proxs=[prox_operators.prox_l1], lr=self.learning_rate, alphas=[self.alpha])
                    self.train_adj(epoch=epoch, i=i, features=data.x, adj=perturbed_adj, data=data)


            if self.logger:
                self.logger.log(metrics)

            if best_metrics is None or (
                metrics['val/loss'] < best_metrics['val/loss'] and
                best_metrics['val/acc'] < metrics['val/acc'] <= metrics['train/maxacc'] and
                best_metrics['train/acc'] < metrics['train/acc'] <= 1.05 * metrics['train/maxacc']
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    break

            # display metrics on progress bar
            epoch_progbar.set_postfix(metrics)

        if self.logger:
            self.logger.log_summary(best_metrics)

        return best_metrics

    def _train(self, data, optimizer, i):
        self.model.train()
        optimizer.zero_grad()
        
        if self.pro and i!=1:
            data = self.update_data(data)

        loss, metrics = self.model.training_step(data)
        loss.backward()
        optimizer.step()
        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        return self.model.validation_step(data)

    def update_data(self, data):
        adj = self.estimator.estimated_adj  #clean graph
        edge_index = adj.nonzero(as_tuple=False).t()
        # package clean adj matrix
        data_ = Data(T=data.T, edge_index=edge_index, test_mask=data.test_mask, train_mask=data.train_mask, val_mask=data.val_mask, x=data.x, y=data.y)
        data_ = ToSparseTensor(remove_edge_index=False)(data_)
        data_.name = data.name
        data_.num_classes = data.num_classes
        return data_.to(self.device)

    def train_adj(self, epoch, i, features, adj, data=None):
        estimator = self.estimator
        t_ = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        
        _, _, p_yt_x = self.model(data)
        loss_gnn = self.model.cross_entropy_loss(p_y=p_yt_x[data.train_mask], y=self.model.cached_yt[data.train_mask], weighted=False)
        
        loss_diffiential =  loss_fro + self.gamma * loss_gnn 
        loss_diffiential.backward()

        self.optimizer_adj.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        # use total loss
        # total_loss = loss_fro + self.gamma * loss_gnn + self.alpha * loss_l1 
        # total_loss.backward()

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        self.model.eval()

        print(
            'Epoch_Pro: {:04d}_{:d}'.format(epoch+1, i),
            'diffiential loss: {:.4f}'.format(loss_diffiential),
            'time: {:.4f}s'.format(time.time() - t_)
        )