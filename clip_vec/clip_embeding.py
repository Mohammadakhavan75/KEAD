# !pip install torch torchvision sentencepiece
# !pip install git+https://github.com/openai/CLIP.git

import torch
from PIL import Image
from torchvision import transforms
from clip import clip
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader
import numpy as np
import PIL
from tqdm import tqdm
import argparse
import pickle

class load_np_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets_path, transform):
        self.data = np.load(imgs_path)
        self.targets = np.load(targets_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , target = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
        if transform:
            img = self.transform(img)

        return img, target

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
    parser.add_argument('--transform', type=int, 
                        default=0, help='use transformation dataset')
    parser.add_argument('--dataset', type=str, 
                        default='cifar10', help='cifar10-cifar100-svhn')

    args = parser.parse_args()
    return args

args = parsing()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)


if args.dataset == 'cifar10':
    cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    noraml_dataset = CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)
    if args.transform:
        cifar_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/CIFAR-10-Train-R-A/{args.aug}.npy'
        cifar_train_cor_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/CIFAR-10-Train-R-A/labels-A.npy'
    else:
        cifar_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/CIFAR-10-Train-R-C/{args.aug}.npy'
        cifar_train_cor_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/CIFAR-10-Train-R-C/labels-C.npy'

    aug_dataset = load_np_dataset(cifar_train_cor_img_path, cifar_train_cor_target_path, transform=transform)

elif args.dataset == 'svhn':
    svhn_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    noraml_dataset = SVHN(root=svhn_path, split="train", transform=transform)
    if args.transform:
        svhn_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/SVHN-Train-R-A/{args.aug}.npy'
        svhn_train_cor_target_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/SVHN-Train-R-A/labels-A.npy'
    else:
        svhn_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/SVHN-Train-R-C/{args.aug}.npy'
        svhn_train_cor_target_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/manual/SVHN-Train-R-C/labels-C.npy'

    aug_dataset = load_np_dataset(svhn_train_cor_img_path, svhn_train_cor_target_path, transform=transform)

else:
    raise NotImplemented(f"Not Available Dataset {args.dataset}!")


normal_loader = DataLoader(noraml_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
aug_loader = DataLoader(aug_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)


loader = zip(normal_loader, aug_loader)
diffs = []
targets_list = []
for i, data in enumerate(tqdm(loader)):
    data_normal, data_aug = data
    imgs_n, targets = data_normal
    imgs_aug, _ = data_aug
    # imgs_n, imgs_aug = transform(imgs_n).to(device), transform(imgs_aug).to(device)
    if len(imgs_n) != len(imgs_aug): # if len imgs_aug was larger than imgs_normal
        imgs_aug = imgs_aug[:len(imgs_n)]
        imgs_n, imgs_aug = imgs_n.to(device), imgs_aug.to(device)
        imgs_n_features = model.encode_image(imgs_n)
        imgs_aug_features = model.encode_image(imgs_aug)
        diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        break

    imgs_n, imgs_aug = imgs_n.to(device), imgs_aug.to(device)
    imgs_n_features = model.encode_image(imgs_n)
    imgs_aug_features = model.encode_image(imgs_aug)
    diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
    targets_list.extend(targets.detach().cpu().numpy())

diffs = np.asarray(diffs)
targets_list = np.asarray(targets_list)
with open(f'./tensors/{args.dataset}_diffs_{args.aug}.pkl', 'wb') as f:
    pickle.dump(diffs, f)
with open(f'./tensors/{args.dataset}_diffs_target_{args.aug}.pkl', 'wb') as f:
    pickle.dump(targets_list, f)


# t, g = zip(*sorted(zip(targets_list, diffs)))
# for i in range(10):
#     print(f"class {i}: {np.mean(g[i*5000:(i+1)*5000])}")


# vals = []
# vals_softmax = []
# for i in range(10):
#     vals.append(torch.tensor(np.mean(g[i*5000:(i+1)*5000])).float())

# vals_softmax = torch.nn.functional.softmax(torch.tensor(vals)).detach().cpu().numpy()

# with open(f'./softmax/{args.dataset}_{args.aug}.out', 'w') as file:
#     for val in vals_softmax:
#         file.write(str(val)+'\n')