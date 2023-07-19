import torch
import torch.nn as nn
import numpy as np
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F


def compute_err_torch(input1, input2, Plogs, log_cof):
    tote = float('inf')
    for k in range(n_iters):
        m1 = F.mse_loss(input1[k], input2)
        er_i = m1 - log_cof * Plogs[k]
        tote = min(tote, er_i)
    return tote

def make_cinit_single(n_inits, class_):
    c_init = np.zeros((n_inits))
    for i in range(n_inits):
        c_init[i] = class_
    return c_init


def compute_err_rand_init(tot_e, logs, log_cof):
    tot_es = tot_e.reshape(-1, 28 * 28).mean(axis=1).reshape((10, 1)) - log_cof * logs.cpu().data.numpy()
    return np.min(tot_es), np.argmin(tot_es)


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def enc_init(E, x, c):
    z_mu, z_var = E(x, c)
    std = torch.exp(z_var / 2)
    eps = torch.randn_like(std)
    Zinit = eps.mul(std).add_(z_mu)
    Zinit = Variable(Zinit.cuda(), requires_grad=True)
    return Zinit


def reparameterize(mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = Variable(logvar.data.new(logvar.size()).normal_())
    return eps.mul(logvar).add_(mu)


def add_noise(inputs):
    noise = torch.randn_like(inputs) * 0.1
    return inputs + noise


def calculate_kld(mean, log_var):
    KLD = -0.5 * torch.sum(-log_var.exp() - torch.pow(mean, 2) + log_var + 1, 1)
    return KLD.mean(0, True)


def idx2onehot(idx, n=10):
    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)
    return onehot


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

# this function found https://github.com/jalola/improved-wgan-pytorch/blob/master/training_utils.py
# Modified by:
# In 2021-05-26
# LICENSE: MIT LICENSE
def calc_gradient_penalty_un(netD, real_data, fake_data, batch_size, dim, device, gp_lambda):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 1, dim, dim)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 1, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

# this function found https://github.com/jalola/improved-wgan-pytorch/blob/master/training_utils.py
# Modified by: mitra alirezaei
# In 2021-05-26
# LICENSE: MIT LICENSE
def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, device, gp_lambda, y_real):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 1, dim, dim)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 1, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, y_real)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty



