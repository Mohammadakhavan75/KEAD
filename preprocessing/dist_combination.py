import os
import json
import torch
import pickle
import argparse
import numpy as np


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    parser.add_argument('--dataset', type=str, 
                        default='cifar10', help='cifar10-cifar100-svhn')
    parser.add_argument('--backbone', type=str, 
                        default='clip', help='clip or dinov2')
    parser.add_argument('--config', type=str, 
                        help='config')
    parser.add_argument('--one_class', action='store_true', 
                        help='config')

    args = parser.parse_args()

    return args


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


def get_target_labels(args, generalization_path):
    
    if args.dataset == 'svhn':
        classes = 10    
        targets_list_loaded = np.load(os.path.join(generalization_path, 'svhn_Train_s1/labels.npy'))

    if args.dataset == 'cifar10':
        classes = 10
        targets_list_loaded = np.load(os.path.join(generalization_path, 'cifar10_Train_s1/labels.npy'))

    if args.dataset == 'cifar100':
        classes = 20
        targets_list_loaded = np.load(os.path.join(generalization_path, 'cifar100_Train_s1/labels.npy'))
        targets_list_loaded = sparse2coarse(targets_list_loaded)
        
    if args.dataset == 'imagenet30':
        classes = 30
        targets_list_loaded = np.load(os.path.join(generalization_path, 'imagenet_Train_s1/labels.npy'))
    
    if args.dataset == 'mvtec_ad':
        classes = 15
        targets_list_loaded = np.load(os.path.join(generalization_path, 'mvtec_ad_Train_s1/labels.npy'))

    return targets_list_loaded, classes

args = parsing()

with open(args.config, 'r') as config_file:
    config = json.load(config_file)

target_labels, classes = get_target_labels(args, config['generalization_path'])

root = f'./wasser_dist/{args.backbone}/{args.dataset}/'
loaded_diffs = {}
for file_name in os.listdir(root):
    with open(os.path.join(root, file_name), 'rb') as f:
        loaded = pickle.load(f)
        loaded_diffs[file_name.split('.')[0]] = loaded

if args.one_class:
    softmax_sorted = {}
    for class_idx in range(classes):
        softmax_sorted[class_idx] = {}

    for class_idx in range(classes):
        indices = [k for k in range(len(target_labels)) if target_labels[k]==class_idx]
        for noise_name in loaded_diffs.keys():
            print(class_idx)
            softmax_sorted[class_idx][noise_name] = torch.mean(torch.tensor([loaded_diffs[noise_name][class_idx] for idx in indices])).float()
            
        # for file_name in os.listdir(root):
        #     with open(os.path.join(root, file_name), 'rb') as f:
        #         loaded_diff = pickle.load(f)
        
            # print(class_idx)
            # noise_name = file_name.split('.')[0]
            # softmax_sorted[class_idx][noise_name] = torch.mean(torch.tensor([loaded_diff[class_idx].astype(np.float32) for idx in indices])).float()
        
    for i in range(classes):
        softmaxes = torch.nn.functional.softmax(torch.tensor(list(softmax_sorted[i].values())), dim=0).numpy()
        for j, key in enumerate(softmax_sorted[i].keys()):
            softmax_sorted[i][key] = softmaxes[j]

        sorted_values = sorted(softmax_sorted[i].items(), key=lambda item: item[1])
        softmax_sorted[i] = dict(sorted_values)

    os.makedirs(f'./ranks/{args.backbone}/{args.dataset}/', exist_ok=True)
    with open(f'./ranks/{args.backbone}/{args.dataset}/wasser_dist_softmaxed.pkl', 'wb') as f:
        pickle.dump(softmax_sorted, f)
else:
    softmax_sorted = {}
    for noise_name in loaded_diffs.keys():
        softmax_sorted[noise_name] = torch.mean(torch.tensor(loaded_diffs[noise_name]))

    softmaxes = torch.nn.functional.softmax(torch.tensor(list(softmax_sorted.values())), dim=0).numpy()
    for j, key in enumerate(softmax_sorted.keys()):
        softmax_sorted[key] = softmaxes[j]
    
    sorted_values = sorted(softmax_sorted.items(), key=lambda item: item[1])
    softmax_sorted = dict(sorted_values)

    os.makedirs(f'./ranks_datasets/{args.backbone}/{args.dataset}/', exist_ok=True)
    with open(f'./ranks_datasets/{args.backbone}/{args.dataset}/wasser_dist_softmaxed.pkl', 'wb') as f:
        pickle.dump(softmax_sorted, f)
