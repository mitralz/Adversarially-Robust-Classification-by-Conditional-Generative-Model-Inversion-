# Adversarially Robust Classification by Conditional Generative Model Inversion

This repository contains the code implementation for the paper "Adversarially Robust Classification by Conditional Generative Model Inversion" available at [https://arxiv.org/abs/2201.04733](https://arxiv.org/abs/2201.04733).

## Requirements

To run the code, you need the following dependencies:

- Python 
- PyTorch
- torchvision
- advertorch
- numpy

## Training Conditional Generative Adversarial Network (cGAN)

To train the cGAN model, use the `traincGan.py` script. This script trains a conditional generative adversarial network on a specific dataset with labels. You need to specify the dataset path, the number of epochs, and other hyperparameters within the script.

```bash
python traincGan.py
```

## Reverse Classification using Trained cGAN

After training the cGAN, you can perform reverse classification using the `ReverseG` class provided in the `classifiying.py` file. This script can be used to classify input images based on the trained cGAN model.

## Training Substitute Model for Blackbox Attack

To train a substitute model for blackbox attack, use the `sub_model_train.py` script. You need to specify the dataset path, and other hyperparameters within the script.

```bash
python sub_model_train.py
```

## Blackbox Attacks

The `BB_attacks.py` script provides functionalities for performing blackbox attacks using the trained substitute model. Currently, it supports FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) attacks. You can specify the type of attack, epsilon value, and other attack parameters within the script.

```bash
python BB_attacks.py --fgsm 
```

## Citation

If you find this work useful, please consider citing the original paper:

```
@inproceedings{alirezaei2022adversarial,
  title={Adversarial Robust Classification by Conditional Generative Model Inversion},
  author={Alirezaei, Mitra and Tasdizen, Tolga},
  booktitle={2022 International Conference on Machine Learning and Cybernetics (ICMLC)},
  pages={152--158},
  year={2022},
  organization={IEEE}
}
```
