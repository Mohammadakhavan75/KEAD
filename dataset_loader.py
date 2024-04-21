import PIL
import torch
import pickle
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data.dataset import Subset

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


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def noise_loader(batch_size=32, num_workers=32, one_class_idx=None, tail_positive=None, tail_negative=None):
    
    np_train_target_path = '/storage/users/makhavan/CSI/finals/datasets/data_aug/CorCIFAR10_train/labels.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/data_aug/CorCIFAR10_test/labels.npy'

    np_train_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/'
    np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/'

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


    with open('./clip_vec/softmax_sorted.pkl', 'rb') as file:
        clip_probs = pickle.load(file)

    print("Creating noises loader")
    train_positives_datasets = []
    train_negetives_datasets = []
    test_positives_datasets = []
    test_negetives_datasets = []
    for i in range(10):
        # creating positive noise loader
        noise = list(clip_probs[i].keys())[0]
        np_train_img_path = np_train_root_path + noise + '.npy'
        train_positives_datasets.append(load_np_dataset(np_train_img_path, np_train_target_path, train_transform, train=True))

        np_test_img_path = np_test_root_path + noise + '.npy'
        test_positives_datasets.append(load_np_dataset(np_test_img_path, np_test_target_path, test_transform, train=False))

        # creating negative noise loader
        noise = list(clip_probs[i].keys())[-1]
        np_train_img_path = np_train_root_path + noise + '.npy'
        train_negetives_datasets.append(load_np_dataset(np_train_img_path, np_train_target_path, train_transform, train=True))

        np_test_img_path = np_test_root_path + noise + '.npy'
        test_negetives_datasets.append(load_np_dataset(np_test_img_path, np_test_target_path, test_transform, train=False))

    print("Noises loader created!")

    if one_class_idx != None:
        train_dataset_positives_one_class = get_subclass_dataset(train_positives_datasets[one_class_idx], one_class_idx)
        train_dataset_negetives_one_class = get_subclass_dataset(train_negetives_datasets[one_class_idx], one_class_idx)
        test_dataset_positives_one_class = get_subclass_dataset(test_positives_datasets[one_class_idx], one_class_idx)
        test_dataset_negetives_one_class = get_subclass_dataset(test_negetives_datasets[one_class_idx], one_class_idx)
        
        if tail_positive:
            print(f"Loading positive with tail {tail_positive}")
            with open(f'./clip_vec/tensors/diffs_{list(clip_probs[one_class_idx].keys())[0]}.pkl', 'rb') as file:
                diffs = pickle.load(file)
                
                i = one_class_idx
                class_diff = diffs[i*5000:i*5000 + 5000] / np.max(diffs[i*5000:i*5000 + 5000])
                class_diff_normalized = (class_diff - np.mean(class_diff)) / np.std(class_diff)
                idices = [i for i, element in enumerate(class_diff_normalized) if element  < np.percentile(class_diff_normalized, tail_positive)]

            train_dataset_positives_one_class = Subset(train_dataset_positives_one_class, idices)



        if tail_negative:
            print(f"Loading negatives with tail {tail_negative}")
            with open(f'./clip_vec/tensors/diffs_{list(clip_probs[one_class_idx].keys())[-1]}.pkl', 'rb') as file:
                diffs = pickle.load(file)
                
                i = one_class_idx
                class_diff = diffs[i*5000:i*5000 + 5000] / np.max(diffs[i*5000:i*5000 + 5000])
                class_diff_normalized = (class_diff - np.mean(class_diff)) / np.std(class_diff)
                idices = [i for i, element in enumerate(class_diff_normalized) if element  > np.percentile(class_diff_normalized, tail_negative)]

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
        for i in range(10):
            train_positives.append(DataLoader(train_positives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            train_negetives.append(DataLoader(train_negetives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            test_positives.append(DataLoader(test_positives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            test_negetives.append(DataLoader(test_negetives_datasets[i], shuffle=False, batch_size=batch_size, num_workers=num_workers))
            
    return train_positives, train_negetives, test_positives, test_negetives


def load_cifar10(cifar10_path, batch_size=32, num_workers=32, one_class_idx=None, tail_normal=None):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=False, transform=test_transform, download=True)

    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        test_data = get_subclass_dataset(test_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    if tail_normal:
        print(f"Loading normal_shuffler with tail {tail_normal}")
        with open(f'./clip_vec/tensors/normal_data.pkl', 'rb') as file:
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

