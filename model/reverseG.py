import torch
import numpy as np
from util.util import *


class ReverseG:
    def __init__(self, n_iters, max_epochs, n_classes, log_cof, lr, batch_z_finder, wb=False):
        '''
        The code represents a class called ReverseG that is responsible for reversing a conditional generator (cG) and
        returning the predicted labels or class error. The purpose of reversing the generator is to classify input
        images based on a pre-trained cG model.
        '''
        self.n_iters = n_iters
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.lr = lr
        self.wb = wb
        self.batch_z_finder = batch_z_finder
        self.log_cof = log_cof

    def forward(self, allinput_x):
        pred_y = np.zeros((allinput_x.shape[0]))
        probs_ = torch.zeros((allinput_x.shape[0], self.n_classes))

        for bb in range(allinput_x.shape[0]):
            input_x = allinput_x[bb].repeat(self.n_iters, 1, 1, 1).cuda()
            mseLoss = torch.empty(size=(self.n_classes,))

            for cc in range(self.n_classes):
                class_number = cc
                cinit = make_cinit_single(self.n_iters, class_number)
                cinit = torch.from_numpy(cinit).long()
                Cinit = idx2onehot(cinit.view(-1, 1)).cuda()

                xRec, trlog = self.batch_z_finder(input_x, Cinit)

                c_err = compute_err_torch(xRec, input_x[0, ...], trlog, self.log_cof)
                mseLoss[cc] = c_err if self.wb else torch.argmin(c_err).long().cuda()

            if self.wb:
                probs_[bb, :] = mseLoss
            else:
                pred_y[bb] = torch.argmin(mseLoss).long().cuda()

        if self.wb:
            return probs_
        else:
            return pred_y
