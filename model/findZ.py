import torch
import torch.nn as nn
from torch.autograd import Variable
from util.util import *


class BatchZFinder:
    def __init__(self, generator, z_dim, n_iters, lr, maxEpochs, log_cof):
        self.z_dim = z_dim
        self.n_iters = n_iters
        self.lr = lr
        self.maxEpochs = maxEpochs
        self.log_cof = log_cof
        self.G = generator

    # this function "find_batch_z" written based on "Inverting the Generator of a Generative Adversarial Network" paper
    # and github: https://github.com/ToniCreswell/InvertingGAN
    def find_batch_z(self, Inputx, c_vec, batchNo=0):
        pdf = torch.distributions.Normal(0, 1)
        Zinit0 = Variable(torch.randn(self.n_iters, self.z_dim).cuda(), requires_grad=True)

        # optimizer
        criterion_mse = nn.MSELoss().cuda()
        # criterion_mse = torch.nn.L1Loss().cuda()
        optZ0 = torch.optim.RMSprop([{'params': Zinit0, 'lr': self.lr}])

        for e in range(self.maxEpochs):
            xHAT0 = self.G(Zinit0, c_vec)
            # each element of Z is independent, so likelihood is a sum of log of elements
            logProb0 = pdf.log_prob(Zinit0).mean(dim=1)
            recLoss0 = criterion_mse(xHAT0, Inputx)
            # total loss
            loss0 = recLoss0 - self.log_cof * logProb0.mean()
            optZ0.zero_grad()
            loss0.backward()
            optZ0.step()

        return xHAT0, logProb0

