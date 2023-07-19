import torch
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from model.models import Generator
from model.findZ import BatchZFinder
from model.sub_models import CNNClassifierB
from model.reverseG import ReverseG
import argparse


class SubModelTrain:
    def __init__(self, oracle, substitute, holdout, holdout_dataloader, tedata_loader, batch_size, lamb):
        self.oracle = oracle
        self.substitute = substitute
        self.holdout = holdout
        self.holdout_dataloader = holdout_dataloader
        self.tedata_loader = tedata_loader
        self.augmentation_iters = 6
        self.epochs_per_aug = 10
        self.batch_size = batch_size
        self.lamb = lamb

    def _clamp(self, adv_x, detach=True):
        adv_x = torch.clamp(adv_x, min=-1.0, max=1.0)
        return adv_x.detach_() if detach else adv_x

    def _jacobian_augmentation(self, prev_x, prev_y):
        bs = self.batch_size
        for i in trange(int(np.ceil(prev_x.size(0) / bs)), desc='Jacobian Augmentation'):
            x = prev_x[i * bs:(i + 1) * bs].cuda()
            x.requires_grad_()
            preds = self.substitute(x)
            score = torch.gather(preds, 1, prev_y[i * bs:(i + 1) * bs].unsqueeze(1).cuda())
            score.sum().backward()
            prev_x[i * bs:(i + 1) * bs].add_(self.lamb * x.grad.sign().cpu())
        return self._clamp(prev_x)

    def _train_sub(self):
        print("Training substitute model...")
        bs = self.batch_size
        net = self.substitute
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        x, y = self.holdout
        for aug_iter in trange(self.augmentation_iters, desc='Augmentation Iterations'):
            net.train()
            for epoch in trange(self.epochs_per_aug, desc='Epochs'):
                indices = np.arange(x.size(0))
                np.random.shuffle(indices)
                for batch in trange(int(np.ceil(len(indices) // bs)), desc='Minibatches'):
                    x_b, y_b = x[batch * bs:(batch + 1) * bs], y[batch * bs:(batch + 1) * bs]
                    x_b, y_b = x_b.cuda(), y_b.cuda()
                    pred = net(x_b)
                    loss = F.cross_entropy(pred, y_b)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Save each model after training
            torch.save(
                {
                    'epoch': aug_iter,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                f'exDir/sub_B/sub_epoch_{aug_iter}.pth'
            )
            sub_test_acc = self._testing_sub(net, self.tedata_loader)
            sub_holdout_acc = self._testing_sub(net, self.holdout_dataloader)
            print("Substitute model test accuracy:", sub_test_acc)
            print("Substitute model holdout accuracy:", sub_holdout_acc)

            net.eval()
            if aug_iter != self.augmentation_iters - 1:
                new_x = self._jacobian_augmentation(x, y)
                new_y = self.oracle.forward(new_x.cuda())  # oracle
                x = torch.cat([x, new_x], dim=0)
                y = torch.cat([y, new_y.cpu()], dim=0)
            print("One round of Jacobian augmentation done.")

    def _testing_sub(self, network, val_data_loader):
        network.eval()
        correct_sub, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_data_loader, desc='Testing Sub')):
                x, y = x.cuda(), y.cuda()
                pred_test = network(x)
                y_test = pred_test.argmax(dim=1, keepdim=True)
                correct_sub += y_test.eq(y.view_as(y_test)).sum().item()
                total += y.size(0)
        return correct_sub / total

def main(args):
    G = Generator(args.z_dim)
    state_dictg = torch.load(args.path_to_g)
    G.load_state_dict(state_dictg['state_dict'])
    G = G.cuda().eval()
    for param in G.parameters():
        param.requires_grad = False

    batch_z_finder = BatchZFinder(G, args.z_dim, args.n_iters, args.lr, args.maxEpochs, args.log_cof)
    defense_cG = ReverseG(args.n_iters, args.maxEpochs, args.n_classes, args.log_cof, args.lr, batch_z_finder)
    holdout = (torch.load(args.holdout_X), torch.load(args.holdout_Y))
    holdout_dataloader = DataLoader(TensorDataset(*holdout), batch_size=args.batch_size)
    test_data_loader = DataLoader(TensorDataset(torch.load(args.test_X), torch.load(args.test_Y)), batch_size=args.batch_size)

    sub = CNNClassifierB().cuda()
    attacker = SubModelTrain(defense_cG, sub, holdout, holdout_dataloader, test_data_loader, args.batch_size, args.lamb)

    start_time = time.time()
    attacker._train_sub()
    print("Attack training completed.")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout_X', type=str, help='Path to the holdout X data')
    parser.add_argument('--holdout_Y', type=str, help='Path to the holdout Y data')
    parser.add_argument('--test_X', type=str, help='Path to the test X data')
    parser.add_argument('--test_Y', type=str, help='Path to the test Y data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lamb', type=float, default=0.1, help='Lambda value - jacob. aug.')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--log_cof', type=float, default=0.5, help='Log coefficient')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of z')
    parser.add_argument('--lr', type=float, default=0.025, help='Learning rate for optimizing z')
    parser.add_argument('--path_to_g', type=str, help='Path to the generator model')
    args = parser.parse_args()

    main(args)
