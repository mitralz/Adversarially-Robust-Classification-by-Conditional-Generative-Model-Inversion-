import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model.models import Generator, Discriminator
from util.util import *
import os


def sample_image(epoch_done, batches_done, G, z_size, output_dir):
    """Saves generated images from 0 to n_classes"""
    c = torch.eye(10, 10).cuda()
    for s_j in range(5):
        sample = G(torch.randn(10, z_size).cuda(), c).cpu()
        save_image(sample.view(10, 1, 28, 28), f'{output_dir}/samples_{epoch_done}.png')


def testing_(i, x_rt, y_rt, G, D, epoch, output_dir, z_size):
    # set the evaluation mode
    if i == 0:
        # save each model after training
        torch.save({'state_dict': G.state_dict()}, f'{output_dir}/generator_epoch_{epoch}.pth')
        torch.save({'state_dict': D.state_dict()}, f'{output_dir}/discriminator_epoch_{epoch}.pth')

    batch_size = x_rt.size(0)
    G.eval()
    D.eval()

    with torch.no_grad():
        z_p = torch.randn(batch_size, z_size).cuda()
        x_pt = G(z_p, y_rt)
        ld_r = D(x_rt, y_rt)
        ld_p = D(x_pt, y_rt)

        # ------------D loss------------------
        loss_D = -torch.mean(ld_r) + torch.mean(ld_p)
        # ------------G losses--------------
        loss_G = -torch.mean(ld_p)

        sample_image(epoch, batches_done=i, G=G, z_size=z_size, output_dir=output_dir)

    return loss_D.item(), loss_G.item()


def train_gan(n_epochs, critic_update, z_size, Batch_size, lr_g, lr_d, gp_lambda, n_classes, output_dir):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(
        './fmnistData',
        train=True,
        download=True,
        transform=transforms_
    )

    test_dataset = datasets.FashionMNIST(
        './fmnistData',
        train=False,
        download=True,
        transform=transforms_
    )

    train_iterator = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=Batch_size)

    G = Generator(z_dim=z_size)
    D = Discriminator()
    # Optimizers
    G_trainer = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    D_trainer = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    G = G.cuda()
    D = D.cuda()

    d_update = 0
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(train_iterator):
            imgs = imgs.cuda()
            labels = idx2onehot(labels.view(-1, 1))
            labels = labels.cuda()
            x_r, y_r = imgs, labels
            batch_size = x_r.shape[0]
            z_p = torch.randn(batch_size, z_size).cuda()
            x_p = G(z_p, y_r)
            ld_r = D(x_r, y_r)
            ld_p = D(x_p, y_r)
            # ------------D training-------------
            l_real = -torch.mean(ld_r) + torch.mean(ld_p)
            grad_penalty = calc_gradient_penalty(D, x_r, x_p, batch_size, 28, device, gp_lambda, y_r)
            loss_D = l_real + grad_penalty

            D_trainer.zero_grad()
            loss_D.backward(retain_graph=True)
            D_trainer.step()

            d_update += 1

            if d_update % critic_update == 0:
                z_p = torch.randn(batch_size, z_size).cuda()
                x_p = G(z_p, y_r)
                ld_p = D(x_p, y_r)
                loss_G = -torch.mean(ld_p)
                G_trainer.zero_grad()
                loss_G.backward()
                G_trainer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch}, Train LossD: {loss_D:.2f}, Train LossGD: {loss_G:.2f}')

        ###################################################### testing #####################################################
        for i, (imgst, labelst) in enumerate(test_iterator):
            imgst = imgst.cuda()
            labelst = idx2onehot(labelst.view(-1, 1))
            labelst = labelst.cuda()
            loss_D_it, loss_Git = testing_(i, imgst, labelst, G, D, epoch, output_dir, z_size)

            if i % 100 == 0:
                print(f'Epoch {epoch}, Test LossD: {loss_D_it:.2f}, Test LossGD: {loss_Git:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--critic_update', type=int, default=5, help='frequency of critic updates')
    parser.add_argument('--z_size', type=int, default=100, help='dimension of the latent space')
    parser.add_argument('--Batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate for the generator')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate for the discriminator')
    parser.add_argument('--gp_lambda', type=float, default=10, help='gradient penalty lambda')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--output_dir', type=str, default='./output', help='output directory path')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_gan(args.n_epochs, args.critic_update, args.z_size, args.Batch_size, args.lr_g, args.lr_d,
              args.gp_lambda, args.n_classes, args.output_dir)
