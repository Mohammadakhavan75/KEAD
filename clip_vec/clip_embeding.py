# !pip install torch torchvision sentencepiece
# !pip install git+https://github.com/openai/CLIP.git

import os
import PIL
import sys
import torch
import random
import pickle
import argparse
import numpy as np
from clip import clip
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from torchvision.datasets import CIFAR10, CIFAR100

sys.path.append('/storage/users/makhavan/CSI/exp10/new_contrastive/')
from contrastive import cosine_similarity


class SVHN(Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"],
        'train_and_extra': [
                ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                 "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
                ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                 "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        if self.split == "train_and_extra":
            self.url = self.split_list[split][0][0]
            self.filename = self.split_list[split][0][1]
            self.file_md5 = self.split_list[split][0][2]
        else:
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        if self.split == "test":
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))
        else:
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']

            if self.split == "train_and_extra":
                extra_filename = self.split_list[split][1][1]
                loaded_mat = sio.loadmat(os.path.join(root, extra_filename))
                self.data = np.concatenate([self.data,
                                                  loaded_mat['X']], axis=3)
                self.targets = np.vstack((self.targets,
                                               loaded_mat['y']))
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        if self.split == "test":
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == "test":
            return len(self.data)
        else:
            return len(self.data)

    def _check_integrity(self):
        root = self.root
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            fpath = os.path.join(root, self.filename)
            train_integrity = check_integrity(fpath, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            fpath = os.path.join(root, extra_filename)
            return check_integrity(fpath, md5) and train_integrity
        else:
            md5 = self.split_list[self.split][2]
            fpath = os.path.join(root, self.filename)
            return check_integrity(fpath, md5)

    def download(self):
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            download_url(self.url, self.root, self.filename, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            download_url(self.url, self.root, extra_filename, md5)
        else:
            md5 = self.split_list[self.split][2]
            download_url(self.url, self.root, self.filename, md5)



class load_np_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets_path, transform, train=True):
        if train:
            self.data = np.load(imgs_path)[:50000]
            self.targets = np.load(targets_path)[:50000]
        else:
            self.data = np.load(imgs_path)[:10000]
            self.targets = np.load(targets_path)[:10000]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , target = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
        img = self.transform(img)

        return img, target


class load_np_dataset_SVHN(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets_path, transform, len_data, train=True):
        if train:
            self.data = np.load(imgs_path)[:len_data]
            self.targets = np.load(targets_path)[:len_data]
        else:
            self.data = np.load(imgs_path)[:len_data]
            self.targets = np.load(targets_path)[:len_data]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , target = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
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
    parser.add_argument('--dataset', type=str, 
                        default='cifar10', help='cifar10-cifar100-svhn')
    parser.add_argument('--save_rep_norm', action='store_true',
                        help='cifar10-cifar100-svhn')
    parser.add_argument('--save_rep_aug', action='store_true',
                        help='cifar10-cifar100-svhn')

    args = parser.parse_args()
    return args



args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)
if args.save_rep_norm:
    os.makedirs(f'./representations/{args.dataset}/', exist_ok=True)
    if args.save_rep_aug:
        os.makedirs(f'./representations/{args.dataset}/{args.aug}/', exist_ok=True)

if args.dataset == 'cifar10':
    cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    cifar_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/{args.aug}.npy'
    cifar_train_cor_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/labels_train.npy'
    
    noraml_dataset = CIFAR10(root=cifar10_path, train=True, transform=transform)
    aug_dataset = load_np_dataset(cifar_train_cor_img_path, cifar_train_cor_target_path, transform=transform)
    
elif args.dataset == 'svhn':
    svhn_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    svhn_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Train_AC/{args.aug}.npy'
    svhn_train_cor_target_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Train_AC/labels_train.npy'
    
    noraml_dataset = SVHN(root=svhn_path, split="train", transform=transform)
    aug_dataset = load_np_dataset_SVHN(svhn_train_cor_img_path, svhn_train_cor_target_path, transform=transform, len_data=len(noraml_dataset))

elif args.dataset == 'cifar100':
    cifar100_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    cifar_train_cor_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR100_Train_AC/{args.aug}.npy'
    cifar_train_cor_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR100_Train_AC/labels_train.npy'
    
    noraml_dataset = CIFAR100(root=cifar100_path, train=True, transform=transform)
    aug_dataset = load_np_dataset(cifar_train_cor_img_path, cifar_train_cor_target_path, transform=transform)
    
else:
    raise NotImplemented(f"Not Available Dataset {args.dataset}!")


normal_loader = DataLoader(noraml_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
aug_loader = DataLoader(aug_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)


loader = zip(normal_loader, aug_loader)
cosine_diff = []
wasser_diff = []
diffs = []
targets_list = []
for i, data in enumerate(loader):
    data_normal, data_aug = data
    imgs_n, targets = data_normal
    imgs_aug, _ = data_aug
    # imgs_n, imgs_aug = transform(imgs_n).to(device), transform(imgs_aug).to(device)
    if len(imgs_n) != len(imgs_aug): # if len imgs_aug was larger than imgs_normal
        imgs_aug = imgs_aug[:len(imgs_n)]
        imgs_n, imgs_aug = imgs_n.to(device), imgs_aug.to(device)
        imgs_n_features = model.encode_image(imgs_n)
        imgs_aug_features = model.encode_image(imgs_aug)
        # New impelementation
        if args.save_rep_norm:
            with open(f'./representations/{args.dataset}/batch_{i}.pkl', 'wb') as f:
                pickle.dump(imgs_n_features, f)
        if args.save_rep_aug:
            with open(f'./representations/{args.dataset}/{args.aug}/batch_{i}.pkl', 'wb') as f:
                pickle.dump(imgs_aug_features, f)

        for f_n, f_a in zip(imgs_n_features, imgs_aug_features):
            cosine_diff.append(cosine_similarity(f_n, f_a).detach().cpu().numpy())
            wasser_diff.append(wasserstein_distance(f_n.detach().cpu().numpy(), f_a.detach().cpu().numpy()))

        diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        break

    imgs_n, imgs_aug = imgs_n.to(device), imgs_aug.to(device)
    imgs_n_features = model.encode_image(imgs_n)
    imgs_aug_features = model.encode_image(imgs_aug)

    # New impelementation
    if args.save_rep_norm:
        with open(f'./representations/{args.dataset}/batch_{i}.pkl', 'wb') as f:
            pickle.dump(imgs_n_features, f)
    if args.save_rep_aug:
        with open(f'./representations/{args.dataset}/{args.aug}/batch_{i}.pkl', 'wb') as f:
            pickle.dump(imgs_aug_features, f)

    for f_n, f_a in zip(imgs_n_features, imgs_aug_features):
        cosine_diff.append(cosine_similarity(f_n, f_a).detach().cpu().numpy())
        wasser_diff.append(wasserstein_distance(f_n.detach().cpu().numpy(), f_a.detach().cpu().numpy()))

    diffs.extend(torch.sum(torch.pow((imgs_n_features - imgs_aug_features), 2), dim=1).float().detach().cpu().numpy())
    targets_list.extend(targets.detach().cpu().numpy())

diffs = np.asarray(diffs)
targets_list = np.asarray(targets_list)



os.makedirs(f'./saved_pickles/{args.dataset}/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.dataset}/diffs/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.dataset}/targets/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.dataset}/cosine/', exist_ok=True)
os.makedirs(f'./saved_pickles/{args.dataset}/wasser/', exist_ok=True)


with open(f'./saved_pickles/{args.dataset}/diffs/{args.aug}.pkl', 'wb') as f:
    pickle.dump(diffs, f)
with open(f'./saved_pickles/{args.dataset}/targets/{args.aug}.pkl', 'wb') as f:
    pickle.dump(targets_list, f)
with open(f'./saved_pickles/{args.dataset}/cosine/{args.aug}.pkl', 'wb') as f:
    pickle.dump(cosine_diff, f)
with open(f'./saved_pickles/{args.dataset}/wasser/{args.aug}.pkl', 'wb') as f:
    pickle.dump(wasser_diff, f)


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