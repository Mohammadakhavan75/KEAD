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

from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import wide_resnet50_2
import sys
from tqdm import tqdm

# from utils import CustomImageDataset



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
    parser.add_argument('--config', type=str, 
                        help='config')
    parser.add_argument('--severity', default=1, type=int,
                        help='severity of augmentation selection')
    parser.add_argument('--one_class_idx', default=1, type=int,
                        help='severity of augmentation selection')
    
    args = parser.parse_args()
    return args


args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

with open(args.config, 'r') as config_file:
    config = json.load(config_file)

root_path = config['root_path']
data_path = config['data_path']
imagenet_path = config['imagenet_path']
generalization_path = config['generalization_path']
representations_path = config['representations_path']
args.config = config
args.one_class_idx = None

cosine_diff = []
wasser_diff = []
euclidean_diffs = []
targets_list = []

sys.path.append(args.config["library_path"])
from dataset_loader import get_loader
from models.utils import augmentation_layers as augl

if args.gpu != str(0):
    device = f'cuda:{args.gpu}'
else:
    device = 'cuda'


if args.backbone == 'clip':
    model, transform = clip.load("ViT-L/14", device=device)
    # model, transform = clip.load("RN50x4", device=device)
    model = model.to(device)
elif args.backbone == 'resnet50':
    model = wide_resnet50_2(weights="IMAGENET1K_V2")
    layers = list(model.children())
    layers = layers[:-1] # Remove the last layer
    model = torch.nn.Sequential(*layers) # Reconstruct the model without the last layer
    model = model.to(device)
    if args.dataset != 'imagenet':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((224,224))])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224)])
elif args.backbone == 'dinov2':
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model = model.to(device)
    if args.dataset != 'imagenet':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((224,224))])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224)])
else:
    raise NotImplemented("Backbone not available !!!")


if args.save_rep_norm:
    rep_norm_path = os.path.normpath(f'{representations_path}/{args.backbone}/{args.dataset}/seed_{args.seed}/normal/').replace("\r", "")
    os.makedirs(rep_norm_path, exist_ok=True)
if args.save_rep_aug:
    rep_aug_path = os.path.normpath(f'{representations_path}/{args.backbone}/{args.dataset}/seed_{args.seed}/{args.aug}/').replace("\r", "")
    os.makedirs(rep_aug_path, exist_ok=True)

save_pickles_path = os.path.normpath(f'./{representations_path}/{args.backbone}/{args.dataset}/seed_{args.seed}').replace("\r", "")

os.makedirs(f'{save_pickles_path}/targets/'.replace("\r", ""), exist_ok=True)
os.makedirs(f'{save_pickles_path}/euclidean_diffs/'.replace("\r", ""), exist_ok=True)
os.makedirs(f'{save_pickles_path}/cosine_pair/'.replace("\r", ""), exist_ok=True)
os.makedirs(f'{save_pickles_path}/wasser_pair/'.replace("\r", ""), exist_ok=True)


loader, _ = get_loader(args, data_path, imagenet_path, transform)

aug_list = augl.get_augmentation_list()

for aug_name in aug_list:
    if aug_name.lower() == args.aug.lower().replace('_', ''):
        print(f"Using {aug_name} as an augmentation")
        transform_layer = augl.return_aug(aug_name, severity=args.severity).to(device)

to01 = augl.To01(mode='auto')
from01 = augl.From01(mode='auto')

with torch.no_grad():
    model.eval()

    for i, data in tqdm(enumerate(loader)):
        imgs_n, n_targets = data
        imgs_n = imgs_n.to(device)

        if args.backbone == 'clip':
            imgs_to01 = to01(imgs_n)
            imgs_aug = transform_layer(imgs_to01)
        else:
            imgs_aug = transform_layer(imgs_n)

        if args.backbone == 'clip':
            imgs_aug =  from01(imgs_aug)

        if args.backbone == 'clip':
            imgs_n_features = model.encode_image(imgs_n)
            imgs_aug_features = model.encode_image(imgs_aug)
        elif args.backbone == 'dinov2':
            imgs_n_features = model(imgs_n)
            imgs_aug_features = model(imgs_aug)
        elif args.backbone == 'resnet50':
            imgs_n_features = model(imgs_n)
            imgs_aug_features = model(imgs_aug)


        # Saving the representations
        if args.save_rep_norm:
            with open(os.path.join(rep_norm_path, f'batch_{i}.pkl').replace("\r", ""), 'wb') as f:
                pickle.dump(imgs_n_features.detach().cpu().numpy(), f)
        if args.save_rep_aug:
            with open(os.path.join(rep_aug_path, f'batch_{i}.pkl').replace("\r", ""), 'wb') as f:
                pickle.dump(imgs_aug_features.detach().cpu().numpy(), f)

        # for f_n, f_a in zip(imgs_n_features, imgs_aug_features):
            # wasser_diff.append(wasserstein_distance(f_n.detach().cpu().numpy(), f_a.detach().cpu().numpy()))

        cosine_diff.append(cosine_similarity(imgs_n_features.detach().cpu().numpy(), imgs_aug_features.detach().cpu().numpy()))
        euclidean_diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
        targets_list.extend(n_targets.detach().cpu().numpy())

euclidean_diffs = np.asarray(euclidean_diffs)
targets_list = np.asarray(targets_list)
# cosine_diff = np.asarray(cosine_diff)



with open(f'{save_pickles_path}/targets/{args.aug}.pkl'.replace("\r", ""), 'wb') as f:
    pickle.dump(targets_list, f)
with open(f'{save_pickles_path}/euclidean_diffs/{args.aug}.pkl'.replace("\r", ""), 'wb') as f:
    pickle.dump(euclidean_diffs, f)
with open(f'{save_pickles_path}/cosine_pair/{args.aug}.pkl'.replace("\r", ""), 'wb') as f:
    pickle.dump(cosine_diff, f)
# with open(f'{save_pickles_path}/wasser_pair/{args.aug}.pkl'.replace("\r", ""), 'wb') as f:
#     pickle.dump(wasser_diff, f)
