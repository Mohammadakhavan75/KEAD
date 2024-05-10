import random
import os
import pickle
import ot
import torch
import numpy as np
import argparse


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aug', type=str, default='gaussian_noise',
                        help='select noise.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    parser.add_argument('--dataset', type=str, 
                        default='cifar10', help='cifar10-cifar100-svhn')

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


# with open('/storage/users/makhavan/CSI/exp10/new_contrastive/clip_vec/saved_pickles/cifar10/targets/flip.pkl', 'rb') as file:
#     targets_list_loaded = pickle.load(file)
targets_list_loaded = np.load('/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/labels_train.npy')


for i in range(len(os.listdir(f'./representations/{args.dataset}/normal/'))):
    with open(f'./representations/{args.dataset}/normal/batch_{i}.pkl', 'rb') as f:
        imgs_n_features.append(pickle.load(f).detach().cpu().float())

    with open(f'./representations/{args.dataset}/{args.aug}/batch_{i}.pkl', 'rb') as f:
        imgs_aug_features.append(pickle.load(f).detach().cpu().float())

imgs_n_features, imgs_aug_features = torch.cat(imgs_n_features, dim=0).numpy(), torch.cat(imgs_aug_features, dim=0).numpy()

print(f'Results for: {args.aug}')
for class_idx in range(10):
    indices = [k for k in range(len(targets_list_loaded)) if targets_list_loaded[k]==class_idx]

    imgs_n_features_one_class = np.asarray([imgs_n_features[idx].astype(np.float32) for idx in indices])
    imgs_aug_features_one_class = np.asarray([imgs_aug_features[idx].astype(np.float32) for idx in indices])

    cost_matrix = ot.dist(imgs_n_features_one_class, imgs_aug_features_one_class, metric='euclidean')

    # Compute the EMD
    emd_distance = ot.emd2([], [], cost_matrix, numItermax=500000)

    print(f'{class_idx}: {emd_distance}')
