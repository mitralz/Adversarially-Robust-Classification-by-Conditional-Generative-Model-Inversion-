import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
from model.models import Generator
from model.findZ import BatchZFinder
from model.sub_models import CNNClassifierB
from model.reverseG import ReverseG
from advertorch.attacks import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack
import argparse


class BlackBoxAttack:
    def __init__(self, oracle, substitute, fgsm=False, pgd=False):
        self.oracle = oracle
        self.substitute = substitute
        self.white_box = None
        if fgsm:
            self.white_box = GradientSignAttack(self.substitute, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                                clip_min=-1.0, clip_max=1.0, targeted=False)
        elif pgd:
            self.white_box = PGDAttack(self.substitute, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                       clip_min=-1.0, clip_max=1.0, targeted=False)

    def attack(self, data_loader):
        correct_adv, gan_correct, correct_clean, gan_correct_clean = 0, 0, 0, 0
        total = 0
        for i, (x_cln, true_label) in enumerate(tqdm(data_loader, desc='Attack')):
            x_cln, true_label = x_cln.cuda(), true_label.cuda()
            x_adv = self.white_box.perturb(x_cln, true_label)

            y_adv_gan = self.oracle.forward(x_adv)
            gan_correct += y_adv_gan.eq(true_label).sum().item()

            y_clean_gan = self.oracle.forward(x_cln)
            gan_correct_clean += y_clean_gan.eq(true_label).sum().item()

            with torch.no_grad():
                y_clean = self.substitute(x_cln)
                y_adv = self.substitute(x_adv)

            pred_adv = y_adv.argmax(dim=1, keepdim=True)
            correct_adv += pred_adv.eq(true_label.view_as(pred_adv)).sum().item()

            pred_clean = y_clean.argmax(dim=1, keepdim=True)
            correct_clean += pred_clean.eq(true_label.view_as(pred_clean)).sum().item()

            total += true_label.size(0)

            #rec_saveadv = x_adv.view(x_cln.shape[0], 1, 28, 28)
            #save_image(rec_saveadv[:10, ...], join(imgDir, 'adv_batch' + str(i) + '.png'), normalize=True, nrow=10)

            #rec_save = x_cln.view(x_cln.shape[0], 1, 28, 28)
            #save_image(rec_save[:10, ...], join(imgDir, 'clean_batch' + str(i) + '.png'), normalize=True, nrow=10)
            # if total >= 1500:
            #    break
        return correct_adv / total, gan_correct / total, correct_clean / total


def main(args):
    G = Generator(args.z_dim)
    state_dictg = torch.load(args.path_to_g)
    G.load_state_dict(state_dictg['state_dict'])
    G = G.cuda().eval()
    for param in G.parameters():
        param.requires_grad = False

    batch_z_finder = BatchZFinder(G, args.z_dim, args.n_iters, args.lr, args.maxEpochs, args.log_cof)
    defense_cgan = ReverseG(args.n_iters, args.maxEpochs, args.n_classes, args.log_cof, args.lr, batch_z_finder)
    substitute_model = CNNClassifierB().cuda()
    checkpoint = torch.load(args.path_to_net)
    substitute_model.load_state_dict(checkpoint['model_state_dict'])

    xtest = torch.load(args.test_X)
    ytest = torch.load(args.test_Y)
    test_data_loader = DataLoader(TensorDataset(torch.cat(xtest, dim=0), torch.cat(ytest, dim=0)), batch_size=args.batch_size)

    attacker = BlackBoxAttack(defense_cgan, substitute_model.eval(), fgsm=args.fgsm, pgd=args.pgd)

    print("Attacking begins...")
    acc_sub, acc_gan, acc_sub_clean = attacker.attack(test_data_loader)
    print("Accuracy of reverse-cG method:", acc_gan)
    print("Accuracy of Substitute:", acc_sub)
    print("Accuracy of Substitute (clean):", acc_sub_clean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_g', type=str, help='Path to the generator model')
    parser.add_argument('--path_to_net', type=str, help='Path to the substitue model')
    parser.add_argument('--test_X', type=str, help='Path to the test X data')
    parser.add_argument('--test_Y', type=str, help='Path to the test Y data')
    parser.add_argument('--fgsm', action='store_true', help='Use FGSM attack')
    parser.add_argument('--pgd', action='store_true', help='Use PGD attack')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--log_cof', type=float, default=0.5, help='Log coefficient')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of z')
    parser.add_argument('--lr', type=float, default=0.025, help='Learning rate for optimizing z')

    args = parser.parse_args()

    imgDir = './exDir/sub_B'
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)

    main(args)
