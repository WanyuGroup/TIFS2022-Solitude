import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import scipy.sparse as sp
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EstimateAdj(nn.Module):
	def __init__(self, adj, symmetric=False, device=device):
		super(EstimateAdj, self).__init__()
		n = len(adj)
		self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
		self._init_estimation(adj)
		self.symmetric = symmetric
		self.device = device

	def _init_estimation(self, adj):
		with torch.no_grad():
			n = len(adj)
			self.estimated_adj.data.copy_(adj)

	def forward(self):
		return self.estimated_adj

	def normalize(self):
		adj = self.estimated_adj
		normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(device))
		return normalized_adj

	def _normalize(self, mx):
		rowsum = mx.sum(1)
		r_inv = rowsum.pow(-1/2).flatten()
		r_inv[torch.isinf(r_inv)] = 0.
		r_mat_inv = torch.diag(r_inv)
		mx = r_mat_inv @ mx
		mx = mx @ r_mat_inv
		return mx


class PGD(Optimizer):
    def __init__(self, params, proxs, alphas, lr, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():
    """Proximal Operators.
    """
    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm.
        """
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data)-alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm).
        """
        device = data.device
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):
        device = data.device
        U, S, V = torch.svd(data)
        self.nuclear_norm = S.sum()
        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]),range(0, U.shape[0])]).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V
prox_operators = ProxOperators()