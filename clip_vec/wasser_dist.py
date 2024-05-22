import random
import os
import pickle
import ot
import torch
import numpy as np
import argparse
from scipy.spatial.distance import cosine

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
    parser.add_argument('--dino', action="store_true", help='superclass')

    args = parser.parse_args()
    return args



args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


imgs_n_features = []
imgs_aug_features = []


# with open('/exp10/new_contrastive/clip_vec/saved_pickles/cifar10/targets/flip.pkl', 'rb') as file:
#     targets_list_loaded = pickle.load(file)


if args.super_class:
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
    # Usage example:

    # targets_list_loaded = [i for t in targets_list_loaded for i , sub_list in enumerate(CIFAR100_SUPERCLASS) if t in sub_list]


for i in range(len(os.listdir(f'./representations/{args.dataset}/normal/'))):
    with open(f'./representations/{args.dataset}/normal/batch_{i}.pkl', 'rb') as f:
        imgs_n_features.append(pickle.load(f).detach().cpu().float())

    with open(f'./representations/{args.dataset}/{args.aug}/batch_{i}.pkl', 'rb') as f:
        imgs_aug_features.append(pickle.load(f).detach().cpu().float())

if args.dino:
    for i in range(len(os.listdir(f'./DINO/representations/{args.dataset}/normal/'))):
        with open(f'./DINO/representations/{args.dataset}/normal/batch_{i}.pkl', 'rb') as f:
            imgs_n_features.append(pickle.load(f).detach().cpu().float())

        with open(f'./DINO/representations/{args.dataset}/{args.aug}/batch_{i}.pkl', 'rb') as f:
            imgs_aug_features.append(pickle.load(f).detach().cpu().float())

imgs_n_features, imgs_aug_features = torch.cat(imgs_n_features, dim=0).numpy(), torch.cat(imgs_aug_features, dim=0).numpy()

print(f'Results for: {args.aug}')
if args.dataset == 'svhn':
    classes = 10
    os.makedirs('./wasser_dist/svhn/', exist_ok=True)
    targets_list_loaded = np.load('/finals/datasets/generalization_repo_dataset/SVHN_Train_AC/labels_train.npy')
if args.dataset == 'imagenet30':
    classes = 30
    os.makedirs('./wasser_dist/imagenet30/', exist_ok=True)
    targets_list_loaded = np.load('/finals/datasets/generalization_repo_dataset/Imagenet_Train_AC/labels_train.npy')
if args.dataset == 'cifar10':
    classes = 10
    os.makedirs('./wasser_dist/cifar10/', exist_ok=True)
    targets_list_loaded = np.load('/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/labels_train.npy')
if args.dataset == 'cifar100':
    classes = 100
    os.makedirs('./wasser_dist/cifar100/', exist_ok=True)
    targets_list_loaded = np.load('/finals/datasets/generalization_repo_dataset/CIFAR100_Train_AC/labels_train.npy')
    if args.super_class:
        classes = 20
        targets_list_loaded = np.load('/finals/datasets/generalization_repo_dataset/CIFAR100_Train_AC/labels_train.npy')
        targets_list_loaded = sparse2coarse(targets_list_loaded)
        os.makedirs('./wasser_dist/cifar100_superclass/', exist_ok=True)



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

if args.dataset == 'cifar100':
    if args.super_class:
        with open(f'./wasser_dist/cifar100_superclass/dist_{args.aug}.pkl', 'wb') as f:
            pickle.dump(distances, f)
    else:
        with open(f'/wasser_dist/cifar100/dist_{args.aug}.pkl', 'wb') as f:
            pickle.dump(distances, f)

if args.dataset == 'cifar10':
    with open(f'./wasser_dist/cifar10/dist_{args.aug}.pkl', 'wb') as f:
            pickle.dump(distances, f)
if args.dino:
    os.makedirs('./wasser_dist/DINO/cifar10/', exist_ok=True)
    if args.dataset == 'cifar10':
        with open(f'./wasser_dist/DINO/cifar10/dist_{args.aug}.pkl', 'wb') as f:
                pickle.dump(distances, f)

if args.dataset == 'imagenet30':
    with open(f'./wasser_dist/imagenet30/dist_{args.aug}.pkl', 'wb') as f:
            pickle.dump(distances, f)

if args.dataset == 'svhn':
    with open(f'./wasser_dist/svhn/dist_{args.aug}.pkl', 'wb') as f:
            pickle.dump(distances, f)