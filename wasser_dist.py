import random
import os
import pickle
import ot
import torch
import numpy as np
import argparse
import json
from scipy.spatial.distance import cosine

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4, 1, 14,  8,  0,  6,  7,  7, 18,  3,  
                                3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                                0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                                5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                                16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                                10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                                2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                                16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                                18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aug', type=str, default='gaussian_noise',
                        help='select noise.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    parser.add_argument('--dataset', type=str, 
                        default='cifar10', help='cifar10-cifar100-svhn')
    parser.add_argument('--super_class', action="store_true", help='superclass')
    parser.add_argument('--backbone', type=str, 
                        default='clip', help='clip or dinov2')
    parser.add_argument('--config', type=str, 
                        help='config')

    args = parser.parse_args()
    return args



args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

with open('config.json') as config_file:
    config = json.load(config_file)

imgs_n_features = []
imgs_aug_features = []
generalization_path = config['generalization_path']



for i in range(len(os.listdir(f'./representations/{args.backbone}/{args.dataset}/normal/'))):
    with open(f'./representations/{args.backbone}/{args.dataset}/normal/batch_{i}.pkl', 'rb') as f:
        imgs_n_features.append(pickle.load(f).detach().cpu().float())

    with open(f'./representations/{args.backbone}/{args.dataset}/{args.aug}/batch_{i}.pkl', 'rb') as f:
        imgs_aug_features.append(pickle.load(f).detach().cpu().float())

imgs_n_features, imgs_aug_features = torch.cat(imgs_n_features, dim=0).numpy(), torch.cat(imgs_aug_features, dim=0).numpy()

print(f'Running on: {args.aug}')
os.makedirs(f'./wasser_dist/{args.backbone}/{args.dataset}/', exist_ok=True)


if args.dataset == 'svhn':
    classes = 10    
    targets_list_loaded = np.load(os.path.join(generalization_path, '/SVHN_Train_AC/labels_train.npy'))

if args.dataset == 'cifar10':
    classes = 10
    targets_list_loaded = np.load(os.path.join(generalization_path, '/CIFAR10_Train_AC/labels_train.npy'))

if args.dataset == 'cifar100':
    classes = 100
    targets_list_loaded = np.load(os.path.join(generalization_path, '/CIFAR100_Train_AC/labels_train.npy'))
    if args.super_class:
        classes = 20
        targets_list_loaded = np.load(os.path.join(generalization_path, 'CIFAR100_Train_AC/labels_train.npy'))
        targets_list_loaded = sparse2coarse(targets_list_loaded)
       
if args.dataset == 'imagenet30':
    classes = 30
    targets_list_loaded = np.load(os.path.join(generalization_path, '/Imagenet_Train_AC/labels_train.npy'))


distances = []
for class_idx in range(classes):
    indices = [k for k in range(len(targets_list_loaded)) if targets_list_loaded[k]==class_idx]

    imgs_n_features_one_class = np.asarray([imgs_n_features[idx].astype(np.float64) for idx in indices])
    imgs_aug_features_one_class = np.asarray([imgs_aug_features[idx].astype(np.float64) for idx in indices])
    cost_matrix = ot.dist(imgs_n_features_one_class, imgs_aug_features_one_class, metric=cosine)
    # Compute the EMD
    emd_distance = ot.emd2([], [], cost_matrix, numItermax=200000)
    distances.append(emd_distance)
    print(f'{class_idx}: {emd_distance}')


with open(f'./wasser_dist/{args.backbone}/{args.dataset}/dist_{args.aug}.pkl', 'wb') as f:
    pickle.dump(distances, f)
