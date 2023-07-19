import torch
import os
from util.util import *
from torchvision import datasets, transforms
from model.models import Generator
from model.findZ import BatchZFinder
from model.reverseG import ReverseG
import argparse


def main(args):
    exDir = args.exDir
    if not os.path.exists(exDir):
        os.makedirs(exDir)

    batch_size = args.batch_size
    use_gpu = torch.cuda.is_available()

    transform_ = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    test_iterator = torch.utils.data.DataLoader(
        dataset=datasets.FashionMNIST('data/FashionMNIST', train=False, download=True, transform=transform_),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=use_gpu,
    )

    G = Generator(args.z_dim)
    state_dictg = torch.load(args.path_to_g)
    G.load_state_dict(state_dictg['state_dict'])

    for param in G.parameters():
        param.requires_grad = False

    G = G.cuda().eval()
    batch_z_finder = BatchZFinder(G, args.z_dim, args.n_iters, args.lr, args.maxEpochs, args.log_cof)
    classification_model = ReverseG(args.n_iters, args.maxEpochs, args.n_classes, args.log_cof, args.lr, batch_z_finder)
    correct = 0
    total = 0

    for i, (x, y) in enumerate(test_iterator):
        x = x.cuda()
        guess_y = classification_model.forward(x)
        correct += guess_y.eq(y.cuda()).sum().item()
        total += batch_size

    accuracy = correct / total
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    """
    This script utilizes a pretrained conditional generator and classifies images by reversign the cG. 
    Usage:
        python script.py --exDir ./results --path_to_g ./path/to/generator_model.pth --lr 0.025 --maxEpochs 100
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exDir', type=str, default='exDir', help='Directory for saving results')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--log_cof', type=float, default=0.5, help='Log coefficient')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of z')
    parser.add_argument('--path_to_g', type=str, help='Path to the generator model')
    parser.add_argument('--lr', type=float, default=0.025, help='Learning rate for optimizing z')
    parser.add_argument('--maxEpochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--wb', action='store_true', help='Whitebox attack mode')
    args = parser.parse_args()

    main(args)
