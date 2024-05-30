# !pip install torch torchvision sentencepiece
# !pip install git+https://github.com/openai/CLIP.git

import os
import PIL
import sys
import json
import torch
import random
import pickle
import argparse
import torchvision
import numpy as np
from clip import clip
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from torchvision.datasets import CIFAR10, CIFAR100
from contrastive import cosine_similarity
from dataset_loader import SVHN, load_np_dataset


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aug', type=str, default='gaussian_noise',
                        help='select noise.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='number of workers')
    parser.add_argument('--backbone', type=str, 
                        default='clip', help='clip or dinov2')
    parser.add_argument('--dataset', type=str, 
                        default='cifar10', help='cifar10-cifar100-svhn')
    parser.add_argument('--save_rep_norm', action='store_true',
                        help='saving representation for normal data')
    parser.add_argument('--save_rep_aug', action='store_true',
                        help='saving representation for normal data')
    parser.add_argument('--gpu', default='0', type=str, help='gpu number')
    args = parser.parse_args()
    return args


args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = f'cuda:{args.gpu}'

if args.backbone == 'clip':
    model, transform = clip.load("ViT-L/14", device=device)
elif args.backbone == 'dinov2':
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize(224, 224)])
with open('config.json') as config_file:
    config = json.load(config_file)

root_path = config['root_path']
data_path = config['data_path']
imagenet30_path = config['imagenet30_path']
generalization_path = config['generalization_path']


if args.save_rep_norm:
    os.makedirs(f'./representations/{args.backbone}/{args.dataset}/', exist_ok=True)
if args.save_rep_aug:
    os.makedirs(f'./representations/{args.backbone}/{args.dataset}/{args.aug}/', exist_ok=True)

if args.dataset == 'cifar10':
    train_aug_imgs_path = os.path.join(generalization_path, f'CIFAR10_Train_AC/{args.aug}.npy')
    train_aug_targets_path = os.path.join(generalization_path, 'CIFAR10_Train_AC/labels_train.npy')
    
    noraml_dataset = CIFAR10(root=data_path, train=True, transform=transform)
    aug_dataset = load_np_dataset(train_aug_imgs_path, train_aug_targets_path, transform=transform)
    
elif args.dataset == 'svhn':
    train_aug_imgs_path = os.path.join(generalization_path, f'SVHN_Train_AC/{args.aug}.npy')
    train_aug_targets_path = os.path.join(generalization_path, f'SVHN_Train_AC/labels_train.npy')
    
    noraml_dataset = SVHN(root=data_path, split="train", transform=transform)
    aug_dataset = load_np_dataset(train_aug_imgs_path, train_aug_targets_path, transform=transform, dataset='svhn')

elif args.dataset == 'cifar100':
    train_aug_imgs_path = os.path.join(generalization_path, f'CIFAR100_Train_AC/{args.aug}.npy')
    train_aug_targets_path = os.path.join(generalization_path, 'CIFAR100_Train_AC/labels_train.npy')
    
    noraml_dataset = CIFAR100(root=data_path, train=True, transform=transform)
    aug_dataset = load_np_dataset(train_aug_imgs_path, train_aug_targets_path, transform=transform)

elif args.dataset == 'imagenet30':
    imagenet_path = os.path.join(data_path,'ImageNet')
    train_aug_imgs_path = os.path.join(generalization_path, f'Imagenet_Train_AC/{args.aug}.npy')
    train_aug_targets_path = os.path.join(generalization_path, 'Imagenet_Train_AC/labels_train.npy')
    
    if args.backbone != 'clip':
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()])

    noraml_dataset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)
    aug_dataset = load_np_dataset(train_aug_imgs_path, train_aug_targets_path, transform=transform, dataset='imagenet30')
    
else:
    raise NotImplemented(f"Not Available Dataset {args.dataset}!")


normal_loader = DataLoader(noraml_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
aug_loader = DataLoader(aug_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)


loader = zip(normal_loader, aug_loader)
cosine_diff = []
wasser_diff = []
euclidean_diffs = []
targets_list = []
for i, data in enumerate(loader):
    data_normal, data_aug = data
    imgs_n, targets = data_normal
    imgs_aug, _ = data_aug
    # This condition is for when len imgs_aug is larger than imgs_normal
    if len(imgs_n) != len(imgs_aug): 
        imgs_aug = imgs_aug[:len(imgs_n)]
        imgs_n, imgs_aug = imgs_n.to(device), imgs_aug.to(device)
        imgs_n_features = model.encode_image(imgs_n)
        imgs_aug_features = model.encode_image(imgs_aug)
        # Saving the representations
        if args.save_rep_norm:
            with open(f'./representations/{args.backbone}/{args.dataset}/batch_{i}.pkl', 'wb') as f:
                pickle.dump(imgs_n_features, f)
        if args.save_rep_aug:
            with open(f'./representations/{args.backbone}/{args.dataset}/{args.aug}/batch_{i}.pkl', 'wb') as f:
                pickle.dump(imgs_aug_features, f)

        for f_n, f_a in zip(imgs_n_features, imgs_aug_features):
            cosine_diff.append(cosine_similarity(f_n, f_a).detach().cpu().numpy())
            wasser_diff.append(wasserstein_distance(f_n.detach().cpu().numpy(), f_a.detach().cpu().numpy()))

        euclidean_diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        break

    imgs_n, imgs_aug = imgs_n.to(device), imgs_aug.to(device)
    imgs_n_features = model.encode_image(imgs_n)
    imgs_aug_features = model.encode_image(imgs_aug)

    # Saving the representations
    if args.save_rep_norm:
        with open(f'./representations/{args.backbone}/{args.dataset}/batch_{i}.pkl', 'wb') as f:
            pickle.dump(imgs_n_features, f)
    if args.save_rep_aug:
        with open(f'./representations/{args.backbone}/{args.dataset}/{args.aug}/batch_{i}.pkl', 'wb') as f:
            pickle.dump(imgs_aug_features, f)

    for f_n, f_a in zip(imgs_n_features, imgs_aug_features):
        cosine_diff.append(cosine_similarity(f_n, f_a).detach().cpu().numpy())
        wasser_diff.append(wasserstein_distance(f_n.detach().cpu().numpy(), f_a.detach().cpu().numpy()))

    euclidean_diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
    targets_list.extend(targets.detach().cpu().numpy())

euclidean_diffs = np.asarray(euclidean_diffs)
targets_list = np.asarray(targets_list)



os.makedirs(f'./saved_pickles/{args.backbone}/{args.dataset}/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.backbone}/{args.dataset}/diffs/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.backbone}/{args.dataset}/targets/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.backbone}/{args.dataset}/cosine/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.backbone}/{args.dataset}/wasser/', exist_ok=True)


with open(f'./saved_pickles/{args.backbone}/{args.dataset}/targets/{args.aug}.pkl', 'wb') as f:
    pickle.dump(targets_list, f)
with open(f'./saved_pickles/{args.backbone}/{args.dataset}/diffs/{args.aug}.pkl', 'wb') as f:
    pickle.dump(euclidean_diffs, f)
with open(f'./saved_pickles/{args.backbone}/{args.dataset}/cosine/{args.aug}.pkl', 'wb') as f:
    pickle.dump(cosine_diff, f)
with open(f'./saved_pickles/{args.backbone}/{args.dataset}/wasser/{args.aug}.pkl', 'wb') as f:
    pickle.dump(wasser_diff, f)
