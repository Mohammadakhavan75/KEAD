import os
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from contrastive import contrastive
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from cifar10.model import ResNet18, ResNet34, ResNet50
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score

def to_np(x):
    return x.data.cpu().numpy()

def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='starting epoch from.')
    parser.add_argument('--start_epoch', type=int, 
                        default=0, help='starting epoch from.')
    parser.add_argument('--save_path', type=str, 
                        default=None, help='Path to save files.')
    parser.add_argument('--model_path', type=str, 
                        default=None, help='Path to model to resume training.')
    parser.add_argument('--config', type=str, 
                        default="normal", help='Config of data on training model.')
    parser.add_argument('--device', type=str, 
                        default="cuda", help='cuda or cpu.')
    # Optimizer Config
    parser.add_argument('--optimizer', type=str,
                        default='sgd', help='The initial learning rate.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float,
                        default=5, help='The update rate for learning rate.')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.9, help='The gamma param for updating learning rate.')
    parser.add_argument('--last_lr', type=float,
                        default=0, help='The gamma param for updating learning rate.')
                        
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')
    
    parser.add_argument('--noise', default='gaussian_noise', type=str, help='which noise to be run')
    parser.add_argument('--run_index', default=0, type=int, help='run index')
    
    args = parser.parse_args()

    return args


def train(train_loader, positives, negetives, net, train_global_iter, criterion, optimizer, device, writer):

    print("Traning...")
    net = net.to(device)
    net.train()  # enter train mode

    # track train classification accuracy
    epoch_accuracies = {
        'normal': [],
        'positive': [],
        'negative': []
    }
    epoch_loss = {
        'loss': [],
        'ce': [],
        'contrastive': []
    }
    for normal, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 in zip(train_loader,
                       positives[0], positives[1], positives[2], positives[3], positives[4],
                       positives[5], positives[6], positives[7], positives[8], positives[9],
                       negetives[0], negetives[1], negetives[2], negetives[3], negetives[4],
                       negetives[5], negetives[6], negetives[7], negetives[8], negetives[9]):
        ps = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]]
        ns = [n0[0], n1[0], n2[0], n3[0], n4[0], n5[0], n6[0], n7[0], n8[0], n9[0]]
        
        imgs, labels = normal
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        positive_imgs = torch.zeros_like(imgs)
        negative_imgs = torch.zeros_like(imgs)
        for i, label in enumerate(labels):
            positive_imgs[i] = ps[label][i].to(args.device)
            negative_imgs[i] = ns[label][i].to(args.device)
            # positive_imgs, negative_imgs = np.asarray(positive_imgs), np.asarray(negative_imgs)

        positive_imgs, negative_imgs = positive_imgs.to(args.device), negative_imgs.to(args.device)

        optimizer.zero_grad()
        preds, normal_features = model(imgs, True)
        positive_preds, positive_features = model(positive_imgs, True)
        negative_preds, negative_features = model(negative_imgs, True)

        normal_probs = torch.softmax(preds, dim=1)
        positive_probs = torch.softmax(positive_preds, dim=1)
        negative_probs = torch.softmax(negative_preds, dim=1)

        # Calculate loss contrastive for layer
        loss_contrastive = 0
        for norm_f, pos_f, neg_f in zip(normal_features[-1], positive_features[-1], negative_features[-1]):
            loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, neg_f))

        loss_ce = criterion(preds, labels)
        loss = loss_ce + loss_contrastive

        normal_output_index = torch.argmax(normal_probs, dim=1)
        positive_output_index = torch.argmax(positive_probs, dim=1)
        negative_output_index = torch.argmax(negative_probs, dim=1)
        acc_normal = accuracy_score(list(to_np(normal_output_index)), list(to_np(labels)))
        acc_positive = accuracy_score(list(to_np(positive_output_index)), list(to_np(labels)))
        acc_negative = accuracy_score(list(to_np(negative_output_index)), list(to_np(labels)))

        # Logging section
        epoch_accuracies['normal'].append(acc_normal)
        epoch_accuracies['positive'].append(acc_positive)
        epoch_accuracies['negative'].append(acc_negative)
        epoch_loss['loss'].append(loss)
        epoch_loss['ce'].append(loss_ce)
        epoch_loss['contrastive'].append(loss_contrastive)

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss, train_global_iter)
        writer.add_scalar("Train/loss_ce", loss_ce.item(), train_global_iter)
        writer.add_scalar("Train/loss_contrastive", loss_contrastive, train_global_iter)
        writer.add_scalar("Train/acc_normal", acc_normal, train_global_iter)
        writer.add_scalar("Train/acc_positive", acc_positive, train_global_iter)
        writer.add_scalar("Train/acc_negative", acc_negative, train_global_iter)

        loss.backward()
        optimizer.step()

    return train_global_iter, epoch_loss, epoch_accuracies 


def test(eval_loader, positives, negetives, net, global_eval_iter, criterion, device, writer):

    print("Evaluating...")
    net = net.to(device)
    net.eval()  # enter train mode

    # track train classification accuracy
    epoch_accuracies = {
        'normal': [],
        'positive': [],
        'negative': []
    }
    epoch_loss = {
        'loss': [],
        'ce': [],
        'contrastive': []
    }
    with torch.no_grad():
        for normal, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 in zip(eval_loader,
                       positives[0], positives[1], positives[2], positives[3], positives[4],
                       positives[5], positives[6], positives[7], positives[8], positives[9],
                       negetives[0], negetives[1], negetives[2], negetives[3], negetives[4],
                       negetives[5], negetives[6], negetives[7], negetives[8], negetives[9]):
            ps = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]]
            ns = [n0[0], n1[0], n2[0], n3[0], n4[0], n5[0], n6[0], n7[0], n8[0], n9[0]]
            
            imgs, labels = normal
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            positive_imgs = torch.zeros_like(imgs)
            negative_imgs = torch.zeros_like(imgs)
            for i, label in enumerate(labels):
                positive_imgs[i] = ps[label][i].to(args.device)
                negative_imgs[i] = ns[label][i].to(args.device)
                # positive_imgs, negative_imgs = np.asarray(positive_imgs), np.asarray(negative_imgs)

            positive_imgs, negative_imgs = positive_imgs.to(args.device), negative_imgs.to(args.device)

            preds, normal_features = model(imgs, True)
            positive_preds, positive_features = model(positive_imgs, True)
            negative_preds, negative_features = model(negative_imgs, True)

            normal_probs = torch.softmax(preds, dim=1)
            positive_probs = torch.softmax(positive_preds, dim=1)
            negative_probs = torch.softmax(negative_preds, dim=1)

            # Calculate loss contrastive for layer
            loss_contrastive = 0
            for norm_f, pos_f, neg_f in zip(normal_features[-1], positive_features[-1], negative_features[-1]):
                loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, neg_f))

            loss_ce = criterion(preds, labels)
            loss = loss_ce + loss_contrastive

            normal_output_index = torch.argmax(normal_probs, dim=1)
            positive_output_index = torch.argmax(positive_probs, dim=1)
            negative_output_index = torch.argmax(negative_probs, dim=1)
            acc_normal = accuracy_score(list(to_np(normal_output_index)), list(to_np(labels)))
            acc_positive = accuracy_score(list(to_np(positive_output_index)), list(to_np(labels)))
            acc_negative = accuracy_score(list(to_np(negative_output_index)), list(to_np(labels)))

            # Logging section
            epoch_accuracies['normal'].append(acc_normal)
            epoch_accuracies['positive'].append(acc_positive)
            epoch_accuracies['negative'].append(acc_negative)
            epoch_loss['loss'].append(loss)
            epoch_loss['ce'].append(loss_ce)
            epoch_loss['contrastive'].append(loss_contrastive)

            global_eval_iter += 1
            writer.add_scalar("Evaluation/loss", loss, global_eval_iter)
            writer.add_scalar("Evaluation/loss_ce", loss_ce.item(), global_eval_iter)
            writer.add_scalar("Evaluation/loss_contrastive", loss_contrastive, global_eval_iter)
            writer.add_scalar("Evaluation/acc_normal", acc_normal, global_eval_iter)
            writer.add_scalar("Evaluation/acc_positive", acc_positive, global_eval_iter)
            writer.add_scalar("Evaluation/acc_negative", acc_negative, global_eval_iter)


    return global_eval_iter, epoch_loss, epoch_accuracies


def load_model(args):

    # model = torchvision.models.resnet34()
    model = ResNet18(10)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                momentum=args.momentum,weight_decay=args.decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                                      weight_decay=args.decay)
    else:
        raise NotImplemented("Not implemented optimizer!")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)



    return model, criterion, optimizer, scheduler


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

import PIL
import torch
import numpy as np

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
        img = self.transform(img)

        return img, target


def noise_loader(args):
    noises = [
    'brightness',
    'color_jitter',
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'flip',
    'fog',
    'gaussian_blur',
    'gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'jpeg_compression',
    'motion_blur',
    'pixelate',
    'random_crop',
    'rot270',
    'rot90',
    'saturate',
    'shot_noise',
    'snow',
    'spatter',
    'speckle_noise',
    'zoom_blur',
    ]

    np_train_target_path = '/storage/users/makhavan/CSI/finals/datasets/data_aug/CorCIFAR10_train/labels.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/data_aug/CorCIFAR10_test/labels.npy'

    np_train_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Train_AC/'
    np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/'

    train_dict = {}
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    for noise in noises:
        np_train_img_path = np_train_root_path + noise + '.npy'
        train_dict[noise] = load_np_dataset(np_train_img_path, np_train_target_path, train_transform)

    test_dict = {}
    for noise in noises:
        np_test_img_path = np_test_root_path + noise + '.npy'
        test_dict[noise] = load_np_dataset(np_test_img_path, np_test_target_path, test_transform)

    train_loader_dict = {}
    for noise in noises:
        train_loader_dict[noise] = DataLoader(train_dict[noise], shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    
    test_loader_dict = {}
    for noise in noises:
        test_loader_dict[noise] = DataLoader(test_dict[noise], shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader_dict, test_loader_dict

args = parsing()
torch.manual_seed(args.seed)
model, criterion, optimizer, scheduler = load_model(args)



cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
train_dataset, test_dataset = load_cifar10(cifar10_path)

train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
val_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

print("Start Loading noises")
train_loader_dict, test_loader_dict = noise_loader(args)
print("Loading noises finished!")


import pickle
with open('./clip_vec/softmax_sorted.pkl', 'rb') as file:
    clip_probs = pickle.load(file)

print("Creating noises loader")
train_positives = [train_loader_dict[list(clip_probs[i].keys())[0]] for i in range(10)]
train_negetives = [train_loader_dict[list(clip_probs[i].keys())[-1]] for i in range(10)]

test_positives = [test_loader_dict[list(clip_probs[i].keys())[0]] for i in range(10)]
test_negetives = [test_loader_dict[list(clip_probs[i].keys())[-1]] for i in range(10)]
print("Noises loader created!")


if args.model_path is not None:
    save_path = args.save_path
    model_save_path = save_path + 'models/'
else:
    addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
    save_path = f'./run/exp-' + addr + f"_({args.run_index}_{args.noise})_" + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + f'_{args.optimizer}' + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

writer = SummaryWriter(save_path)

train_global_iter = 0
global_eval_iter = 0
best_acc = 0.0
best_loss = np.inf
args.last_lr = args.learning_rate
for epoch in range(0, args.epochs):
    print('epoch', epoch + 1, '/', args.epochs)
    train_global_iter, epoch_loss, epoch_accuracies = train(train_loader, train_positives, train_negetives, model, train_global_iter, criterion, optimizer, args.device, writer)
    global_eval_iter, eval_loss, eval_acc = test(val_loader, test_positives, test_negetives, model, global_eval_iter, criterion, args.device, writer)

    writer.add_scalar("Train/avg_loss", np.mean(epoch_loss['loss']), epoch)
    writer.add_scalar("Train/avg_loss_ce", np.mean(epoch_loss['ce']), epoch)
    writer.add_scalar("Train/avg_loss_contrastive", np.mean(epoch_loss['contrastive']), epoch)
    writer.add_scalar("Train/avg_acc_normal", np.mean(epoch_accuracies['normal']), epoch)
    writer.add_scalar("Train/avg_acc_positive", np.mean(epoch_accuracies['positive']), epoch)
    writer.add_scalar("Train/avg_acc_negative", np.mean(epoch_accuracies['negative']), epoch)
    writer.add_scalar("Evaluation/avg_loss", np.mean(eval_loss['loss']), epoch)
    writer.add_scalar("Evaluation/avg_loss_ce", np.mean(eval_loss['ce']), epoch)
    writer.add_scalar("Evaluation/avg_loss_contrastive", np.mean(eval_loss['contrastive']), epoch)
    writer.add_scalar("Evaluation/avg_acc_normal", np.mean(eval_acc['normal']), epoch)
    writer.add_scalar("Evaluation/avg_acc_positive", np.mean(eval_acc['positive']), epoch)
    writer.add_scalar("Evaluation/avg_acc_negative", np.mean(eval_acc['negative']), epoch)

    writer.add_scalar("Train/lr", args.last_lr, epoch)

    print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])} Train/avg_acc_normal: {np.mean(epoch_accuracies['normal'])} \
          Evaluation/avg_loss: {np.mean(eval_loss['loss'])} Evaluation/avg_acc_normal: {np.mean(eval_acc['normal'])}")
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path,f'model_params_epoch_{epoch}.pt'))

    if np.mean(eval_loss['loss']) < best_loss:
        best_loss = np.mean(eval_loss['loss'])
        torch.save(model.state_dict(), os.path.join(save_path,'best_params.pt'))
    
    # if np.mean(eval_loss['loss']) > best_loss:
    #     scheduler.step()
    #     args.last_lr = scheduler.get_last_lr()[0]
    
# os.remove(f"./run_counter/{args.noise}.log")