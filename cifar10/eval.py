import os
import PIL
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import ResNet18, ResNet34, ResNet50
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score


class load_np_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets_path, transform):
        self.data = np.load(imgs_path)
        self.targets = np.load(targets_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , targets = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
        if transform:
            img = self.transform(img)

        return img, targets
    

def to_np(x):
    return x.data.cpu().numpy()


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='starting epoch from.')
    parser.add_argument('--model_path', type=str, 
                        default=None, help='Path to model to resume training.')
    parser.add_argument('--device', type=str, 
                        default="cuda", help='cuda or cpu.')
    parser.add_argument('--aug', default=None, type=str, help='which aug to be run')
    parser.add_argument('--transform', default=0, type=int, help='is augmentation a transformation or noise')
    parser.add_argument('--run_index', default=0, type=int, help='run index')
    
    args = parser.parse_args()

    return args


def test(loader, net, criterion, device):

    # print("Evaluating...")
    net = net.to(device)
    net.eval()  # enter train mode

    # track train classification accuracy
    eval_acc = []
    eval_loss = []
    with torch.no_grad():
        # for inputs, targets in tqdm(loader):
        for inputs, targets in loader:
            
            inputs , targets = inputs.to(device) , targets.to(device)

            preds = net(inputs)
            probs = torch.softmax(preds, dim=1)

            loss = criterion(preds, targets)
            
            output_index = torch.argmax(probs, dim=1)
            
            acc = accuracy_score(list(to_np(output_index)), list(to_np(targets)))
            eval_acc.append(acc)
            eval_loss.append(loss.item())


    return eval_loss, eval_acc


def load_model(args):

    # model = torchvision.models.resnet34()
    model = ResNet18(10)
    model.load_state_dict(torch.load(args.model_path))

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return model, criterion


def load_cifar10(cifar10_path):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=False, transform=test_transform, download=True)

    return train_data, test_data


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = torch.utils.data.dataset.Subset(dataset, indices)

    return dataset


args = parsing()
torch.manual_seed(args.seed)
model, criterion = load_model(args)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

cifar10_path = '/storage/users/rkashefi/CSI/finals/datasets/data/'
test_dataset = torchvision.datasets.CIFAR10(cifar10_path, train=False, transform=transform, download=True)

if args.transform:
    cifar_train_cor_img_path = f'/storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-R-A/{args.aug}.npy'
    cifar_train_cor_target_path = '/storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-R-A/labels-A.npy'
    aug_dataset = load_np_dataset(cifar_train_cor_img_path, cifar_train_cor_target_path, transform=transform)
elif args.aug:
    cifar_train_cor_img_path = f'/storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-R-C/{args.aug}.npy'
    cifar_train_cor_target_path = '/storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-R-C/labels-C.npy'
    aug_dataset = load_np_dataset(cifar_train_cor_img_path, cifar_train_cor_target_path, transform=transform)


# full_set_cifar_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
# full_set_aug_loader = DataLoader(aug_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

loaders_cifar10 = {}
loaders_aug = {}
for class_n in range(10):
    subclass = get_subclass_dataset(test_dataset, classes=class_n)
    loaders_cifar10[str(class_n)] = DataLoader(subclass, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    if args.aug:
        subclass = get_subclass_dataset(aug_dataset, classes=class_n)
        loaders_aug[str(class_n)] = DataLoader(subclass, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

avg_acc = []
for class_n in range(10):
    
    if args.aug:
        eval_loss_aug, eval_acc_aug = test(loaders_aug[str(class_n)], model, criterion, args.device)
        avg_acc.append(np.mean(eval_acc_aug))
        print(f"class {class_n} acc: {np.mean(eval_acc_aug)}")
    else:
        eval_loss_cifar10, eval_acc_cifar10 = test(loaders_cifar10[str(class_n)], model, criterion, args.device)
        avg_acc.append(np.mean(eval_acc_cifar10))
        print(f"class {class_n} acc: {np.mean(eval_acc_cifar10)}")

    # print(f"avg_loss normal class {class_n}: {np.mean(eval_loss_cifar10)} avg_acc normal class {class_n}: {np.mean(eval_acc_cifar10)}")
    # print(f"avg_loss {args.aug} class {class_n}: {np.mean(eval_loss_aug)} avg_acc {args.aug} class {class_n}: {np.mean(eval_acc_aug)}")
    # print(f"avg_acc class {class_n}: {np.mean(eval_acc_cifar10)} vs {np.mean(eval_acc_aug)}")
    

print(f"average acc of {args.aug}: {np.mean(avg_acc)}")
    