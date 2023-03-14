import math
import torch
import torch.nn.functional as F
from scipy.special import erf
import copy

class Mechanism:
    def __init__(self, eps, input_range, **kwargs):
        self.eps = eps
        self.alpha, self.beta = input_range

    def __call__(self, x):
        raise NotImplementedError


class MultiBit(Mechanism):
    def __init__(self, *args, m='best', **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def __call__(self, x):
        n, d = x.size()
        if self.m == 'best':
            m = int(max(1, min(d, math.floor(self.eps / 2.18))))
        elif self.m == 'max':
            m = d
        else:
            m = self.m

        # sample features for perturbation
        BigS = torch.rand_like(x).topk(m, dim=1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, BigS, True)
        del BigS

        # perturb sampled features
        em = math.exp(self.eps / m)
        p = (x - self.alpha) / (self.beta - self.alpha)
        p = (p * (em - 1) + 1) / (em + 1)
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)
        del p, t, s

        # unbiase the result
        x_prime = d * (self.beta - self.alpha) / (2 * m)
        x_prime = x_prime * (em + 1) * x_star / (em - 1)
        x_prime = x_prime + (self.alpha + self.beta) / 2
        return x_prime

        

class MatrixRandomizedResponse(Mechanism):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, x):
		ss = torch.ones_like(x, dtype=int)
		x_copy = copy.deepcopy(x).to(torch.bool)
		em = math.exp(self.eps)
		p = ss * em / (em + 1)
		print(1 / (em + 1))
		t = torch.bernoulli(p).to(torch.bool)

		# not xor operation, to get the perturbed matrix
		perturbed_matrix = (~(x_copy^t)).to(torch.float)

		# if you need to know how many edge added in the graph
		# xx = ~x.to(torch.bool)
		# print("Added edge: ", (perturbed_matrix.to(torch.bool) & xx).sum() )

		return perturbed_matrix

supported_feature_mechanisms = {
    'mbm': MultiBit,
}

supported_edge_mechanisms = {
    'mrr': MatrixRandomizedResponse,
}
