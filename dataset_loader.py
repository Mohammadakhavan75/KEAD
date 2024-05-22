import os
import PIL
import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset
import transform_layers as TL
from PIL import Image


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
    
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
    def __init__(self, imgs_path, targets_path, transform, dataset, train=True):
        self.dataset = dataset
        if train and dataset == 'cifar10':
            self.data = np.load(imgs_path)[:50000]
            self.targets = np.load(targets_path)[:50000]
        elif dataset == 'cifar10':
            self.data = np.load(imgs_path)[:10000]
            self.targets = np.load(targets_path)[:10000]
        elif dataset == 'anomaly':
            self.data = np.load(imgs_path)[:10000]
            self.targets = np.load(targets_path)[:10000]
            self.anomaly = np.load('/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_anomaly_labels.npy')[:10000]
        else:
            self.data = np.load(imgs_path)
            self.targets = np.load(targets_path)

        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , target = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
        img = self.transform(img)
        if self.dataset == 'anomaly':
            return img, target, self.anomaly[idx]
        else:
            return img, target


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def noise_loader(batch_size=32, num_workers=32, one_class_idx=None, tail_positive=None, tail_negative=None, dataset='cifar10', preprocessing='clip', multi_neg=False):

    if dataset == 'cifar10':
        np_train_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/labels_train.npy'
        np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/labels_test.npy'
        np_train_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/'
        np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/'
        classes = 10
    elif dataset == 'svhn':
        np_train_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Train_AC/labels_train.npy'
        np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Test_AC/labels_test.npy'
        np_train_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Train_AC/'
        np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Test_AC/'
        classes = 10
    elif dataset == 'cifar100':
        np_train_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR100_Train_AC/labels_train.npy'
        np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR100_Test_AC/labels_test.npy'
        np_train_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR100_Train_AC/'
        np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR100_Test_AC/'
        classes = 20


    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    if preprocessing == 'clip':
        with open(f'./ranks/selected_noises_{dataset}_softmaxed.pkl', 'rb') as file:
            clip_probs = pickle.load(file)
    elif preprocessing == 'cosine':
        with open(f'./ranks/{dataset}_cosine_softmaxed.pkl', 'rb') as file:
            clip_probs = pickle.load(file)
    elif preprocessing == 'wasser':
        with open(f'./ranks/{dataset}_wasser.pkl', 'rb') as file:
            clip_probs = pickle.load(file)

    print("Creating noises loader")
    train_positives_datasets = []
    train_negetives_datasets = []
    train_negetives_datasets2 = []
    test_positives_datasets = []
    test_negetives_datasets = []
    for i in range(classes):
        # creating positive noise loader
        noise = list(clip_probs[i].keys())[0]
        print(f"Selecting {noise} for positive pair for class {i}")
        np_train_img_path = np_train_root_path + noise + '.npy'
        train_positives_datasets.append(load_np_dataset(np_train_img_path, np_train_target_path, train_transform, dataset, train=True))

        np_test_img_path = np_test_root_path + noise + '.npy'
        test_positives_datasets.append(load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset, train=False))

        # creating negative noise loader
        if multi_neg:
            noise = list(clip_probs[i].keys())[-1]
            print(f"Selecting {noise} for negetive pair for class {i}")
            np_train_img_path = np_train_root_path + noise + '.npy'
            train_negetives_datasets.append(load_np_dataset(np_train_img_path, np_train_target_path, train_transform, dataset, train=True))
            noise = list(clip_probs[i].keys())[-2]
            print(f"Selecting {noise} for negetive pair2 for class {i}")
            np_train_img_path = np_train_root_path + noise + '.npy'
            train_negetives_datasets2.append(load_np_dataset(np_train_img_path, np_train_target_path, train_transform, dataset, train=True))
            np_test_img_path = np_test_root_path + noise + '.npy'
            test_negetives_datasets.append(load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset, train=False))
        else:
            noise = list(clip_probs[i].keys())[-1]
            print(f"Selecting {noise} for negetive pair for class {i}")
            np_train_img_path = np_train_root_path + noise + '.npy'
            train_negetives_datasets.append(load_np_dataset(np_train_img_path, np_train_target_path, train_transform, dataset, train=True))

            np_test_img_path = np_test_root_path + noise + '.npy'
            test_negetives_datasets.append(load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset, train=False))

    print("Noises loader created!")

    if one_class_idx != None:
        train_dataset_positives_one_class = get_subclass_dataset(train_positives_datasets[one_class_idx], one_class_idx)
        train_dataset_negetives_one_class = get_subclass_dataset(train_negetives_datasets[one_class_idx], one_class_idx)
        test_dataset_positives_one_class = get_subclass_dataset(test_positives_datasets[one_class_idx], one_class_idx)
        test_dataset_negetives_one_class = get_subclass_dataset(test_negetives_datasets[one_class_idx], one_class_idx)
        if multi_neg:
            train_dataset_negetives2_one_class = get_subclass_dataset(train_negetives_datasets2[one_class_idx], one_class_idx)
            train_negetives2 = DataLoader(train_dataset_negetives2_one_class, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        if tail_positive:
            print(f"Loading positive with tail {tail_positive}")
            
            
            if preprocessing == 'clip':
                target_list = np.load(np_train_target_path)
                indices = [k for k in range(len(target_list)) if target_list[k]==one_class_idx]
                with open(f'./clip_vec/tensors_selection/{dataset}_diffs_{list(clip_probs[one_class_idx].keys())[0]}.pkl', 'rb') as file:
                    diffs = pickle.load(file)
                diffs_vals = [diffs[idx] for idx in indices]

            elif preprocessing == 'cosine':
                with open(f'./{dataset}_cosine_totals.pkl', 'rb') as file:
                    diffs = pickle.load(file)
                    diffs_vals = diffs[one_class_idx][list(clip_probs[one_class_idx].keys())[0]]

            elif preprocessing == 'wasser':
                with open(f'./ranks/{dataset}_wasser.pkl', 'rb') as file:
                    diffs = pickle.load(file)
                    diffs_vals = diffs[one_class_idx][list(clip_probs[one_class_idx].keys())[0]]
            
            class_diff = diffs_vals / np.max(diffs_vals)
            class_diff_normalized = (class_diff - np.mean(class_diff)) / np.std(class_diff)
            idices = [j for j, element in enumerate(class_diff_normalized) if element  < np.percentile(class_diff_normalized, tail_positive)]

            train_dataset_positives_one_class = Subset(train_dataset_positives_one_class, idices)

        if tail_negative:
            print(f"Loading negatives with tail {tail_negative}")
            if preprocessing == 'clip':
                target_list = np.load(np_train_target_path)
                indices = [k for k in range(len(target_list)) if target_list[k]==one_class_idx]
                with open(f'./clip_vec/tensors_selection/{dataset}_diffs_{list(clip_probs[one_class_idx].keys())[-1]}.pkl', 'rb') as file:
                    diffs = pickle.load(file)
                diffs_vals = [diffs[idx] for idx in indices]

            elif preprocessing == 'cosine':
                with open(f'./{dataset}_cosine_totals.pkl', 'rb') as file:
                    diffs = pickle.load(file)
                    diffs_vals = diffs[one_class_idx][list(clip_probs[one_class_idx].keys())[-1]]
                    
            
            class_diff = diffs_vals / np.max(diffs_vals)
            class_diff_normalized = (class_diff - np.mean(class_diff)) / np.std(class_diff)
            idices = [j for j, element in enumerate(class_diff_normalized) if element  > np.percentile(class_diff_normalized, tail_negative)]

            train_dataset_negetives_one_class = Subset(train_dataset_negetives_one_class, idices)

        train_positives = DataLoader(train_dataset_positives_one_class, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        train_negetives = DataLoader(train_dataset_negetives_one_class, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        test_positives = DataLoader(test_dataset_positives_one_class, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        test_negetives = DataLoader(test_dataset_negetives_one_class, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    else:
        train_positives = []
        test_positives = []
        train_negetives = []
        test_negetives = []
        for i in range(classes):
            train_positives.append(DataLoader(train_positives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            train_negetives.append(DataLoader(train_negetives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            test_positives.append(DataLoader(test_positives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            test_negetives.append(DataLoader(test_negetives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
    if multi_neg:
        return train_positives, train_negetives, train_negetives2, test_positives, test_negetives
    else:
        return train_positives, train_negetives, test_positives, test_negetives


def load_cifar10(cifar10_path, batch_size=32, num_workers=32, one_class_idx=None, tail_normal=None):

    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # train_transform = transforms.Compose([transforms.ToTensor()])
    # test_transform = transforms.Compose([transforms.ToTensor()])

    s = 1
    size = 32
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])


    train_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=True, transform=data_transforms, download=True)
    test_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=False, transform=data_transforms, download=True)

    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        test_data = get_subclass_dataset(test_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    if tail_normal:
        print(f"Loading normal_shuffler with tail {tail_normal}")
        with open(f'./clip_vec/tensors_selection/cifar10_normal_data.pkl', 'rb') as file:
            diffs = pickle.load(file)            
            i = one_class_idx
            class_diff = diffs[i*5000:i*5000 + 5000] / np.max(diffs[i*5000:i*5000 + 5000])
            class_diff_normalized = (class_diff - np.mean(class_diff)) / np.std(class_diff)
            idices = [i for i, element in enumerate(class_diff_normalized) if element  < np.percentile(class_diff_normalized, tail_normal)]
            normal_data = Subset(train_data, idices)
            normal_loader = DataLoader(normal_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    else:
        normal_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, normal_loader


def load_svhn(svhn_path, batch_size=32, num_workers=32, one_class_idx=None, tail_normal=None):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform = transforms.Compose([transforms.ToTensor()])
    print('loading SVHN')
    train_data = SVHN(root=svhn_path, split="train", transform=transform)

    test_data = SVHN(root=svhn_path, split="test", transform=transform)

    
    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        test_data = get_subclass_dataset(test_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    if tail_normal:
        print(f"Loading normal_shuffler with tail {tail_normal}")
        with open(f'./clip_vec/tensors_selection/svhn_normal_data.pkl', 'rb') as file:
            diffs = pickle.load(file)            
            i = one_class_idx
            class_diff = diffs[i*5000:i*5000 + 5000] / np.max(diffs[i*5000:i*5000 + 5000])
            class_diff_normalized = (class_diff - np.mean(class_diff)) / np.std(class_diff)
            idices = [i for i, element in enumerate(class_diff_normalized) if element  < np.percentile(class_diff_normalized, tail_normal)]
            normal_data = Subset(train_data, idices)
            normal_loader = DataLoader(normal_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    else:
        normal_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, normal_loader


def load_cifar100(path, batch_size=32, num_workers=32, one_class_idx=None):

    CIFAR100_SUPERCLASS = [
        [4, 31, 55, 72, 95],
        [1, 33, 67, 73, 91],
        [54, 62, 70, 82, 92],
        [9, 10, 16, 29, 61],
        [0, 51, 53, 57, 83],
        [22, 25, 40, 86, 87],
        [5, 20, 26, 84, 94],
        [6, 7, 14, 18, 24],
        [3, 42, 43, 88, 97],
        [12, 17, 38, 68, 76],
        [23, 34, 49, 60, 71],
        [15, 19, 21, 32, 39],
        [35, 63, 64, 66, 75],
        [27, 45, 77, 79, 99],
        [2, 11, 36, 46, 98],
        [28, 30, 44, 78, 93],
        [37, 50, 65, 74, 80],
        [47, 52, 56, 59, 96],
        [8, 13, 48, 58, 90],
        [41, 69, 81, 85, 89],
    ]

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=transform)
    
    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, CIFAR100_SUPERCLASS[one_class_idx])
        test_data = get_subclass_dataset(test_data, CIFAR100_SUPERCLASS[one_class_idx])

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    normal_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, normal_loader


def csi_aug():
    # Align augmentation
    image_size = (32, 32, 3)
    resize_scale = (0.54, 1)
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )
    return transform

