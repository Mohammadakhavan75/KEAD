import os
import json
import torch
import random
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from utils import tsne_plot
from datetime import datetime
from contrastive import contrastive
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from cifar10.model import ResNet18, ResNet34, ResNet50
from cifar10.models.resnet_imagenet import resnet18, resnet50
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset_loader import noise_loader, load_cifar10, load_svhn, load_cifar100, load_imagenet30

from dataset_loader import load_np_dataset, get_subclass_dataset
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px 


def to_np(x):
    return x.data.cpu().numpy()


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int, 
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
                        default=3, help='The update rate for learning rate.')
    parser.add_argument('--milestones', nargs='+', type=int, default=[500, 800],
                         help='A list of milestones')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.1, help='The gamma param for updating learning rate.')
    parser.add_argument('--last_lr', type=float,
                        default=0, help='The gamma param for updating learning rate.')
                        
    parser.add_argument('--lamb', type=float, default=1, help='loss scaling parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')

    parser.add_argument('--run_index', default=0, type=int, help='run index')

    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10-cifar100-svhn')
    parser.add_argument('--one_class_idx', default=None, type=int, help='select one class index')
    parser.add_argument('--temperature', default=0.5, type=float, help='chaning temperature of contrastive loss')
    parser.add_argument('--preprocessing', default='clip', type=str, help='which preprocessing use for noise order')
    parser.add_argument('--shift_normal', action="store_true", help='A list of milestones')
    parser.add_argument('--device_num', default='0', type=str, help='A list of milestones')
    parser.add_argument('--csi_aug', action="store_true", help='A list of milestones')
    parser.add_argument('--k_pairs', default=1, type=int, help='A list of milestones')
    parser.add_argument('--gpu', default=0, type=int, help='A list of milestones')
    parser.add_argument('--e_holder', default=0, type=str, help='A list of milestones')
    parser.add_argument('--tsne', action="store_true", help='A list of milestones')
    parser.add_argument('--linear', action="store_true", help='A list of milestones')
    parser.add_argument('--config', default=None, help='config file')
    args = parser.parse_args()

    return args


def train_one_class(train_loader, train_positives_loader, train_negetives_loader,
                     net, train_global_iter, optimizer, device, writer):

    # Enter model into train mode
    net = net.to(device)
    net.train()
    
    # Tracking metrics
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
    
    sim_ps, sim_ns = [], []
    
    if args.k_pairs == 1: 
        for normal, p_data, n_data in zip(train_loader, train_positives_loader[0], train_negetives_loader[0]):
            imgs, labels = normal
            p_imgs, _ = p_data
            n_imgs, _ = n_data
            
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            p_imgs = p_imgs.to(args.device)
            n_imgs = n_imgs.to(args.device)
        
        
            optimizer.zero_grad()
            _, normal_features = model(imgs, True)
            _, p_features = model(p_imgs, True)
            _, n_features = model(n_imgs, True)

            normal_features = normal_features[-1]
            p_features = p_features[-1]
            n_features = n_features[-1]

            loss_contrastive = torch.tensor(0.0, requires_grad=True)

            for norm_f, p_f, n_f  in zip(normal_features, p_features, n_features):
                l_c, sim_p, sim_n = contrastive(norm_f, p_f, n_f, temperature=args.temperature)
                loss_contrastive = loss_contrastive + l_c
                sim_ps.append(sim_p.detach().cpu())
                sim_ns.append(sim_n.detach().cpu())
            
        
            loss = loss_contrastive / len(normal_features)
            
            epoch_loss['loss'].append(loss.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            train_global_iter += 1
            writer.add_scalar("Train/loss", loss.item(), train_global_iter)
            writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)

            loss.backward()
            optimizer.step()

    elif args.k_pairs == 2:
        for normal, p_data, n_data, p_data1, n_data1 in zip(train_loader, train_positives_loader[0], train_negetives_loader[0], train_positives_loader[1], train_negetives_loader[1]):
            imgs, labels = normal
            p_imgs, _ = p_data
            n_imgs, _ = n_data
            p_imgs1, _ = p_data1
            n_imgs1, _ = n_data1
            
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            p_imgs = p_imgs.to(args.device)
            n_imgs = n_imgs.to(args.device)
            p_imgs1 = p_imgs1.to(args.device)
            n_imgs1 = n_imgs1.to(args.device)
        
        
            optimizer.zero_grad()
            _, normal_features = model(imgs, True)
            _, p_features = model(p_imgs, True)
            _, n_features = model(n_imgs, True)
            _, p_features1 = model(p_imgs1, True)
            _, n_features1 = model(n_imgs1, True)

            normal_features = normal_features[-1]
            p_features = p_features[-1]
            n_features = n_features[-1]
            p_features1 = p_features1[-1]
            n_features1 = n_features1[-1]

            loss_contrastive = torch.tensor(0.0, requires_grad=True)

            for norm_f, p_f, n_f, p_f1, n_f1  in zip(normal_features, p_features, n_features, p_features1, n_features1):
                p_fs = torch.stack([p_f, p_f1], dim=0)
                n_fs = torch.stack([n_f, n_f1], dim=0)
                l_c, sim_p, sim_n = contrastive(norm_f, p_fs, n_fs, temperature=args.temperature)
                loss_contrastive = loss_contrastive + l_c
                sim_ps.append(sim_p.detach().cpu())
                sim_ns.append(sim_n.detach().cpu())
            
        
            loss = loss_contrastive / len(normal_features)
            
            epoch_loss['loss'].append(loss.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            train_global_iter += 1
            writer.add_scalar("Train/loss", loss.item(), train_global_iter)
            writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)

            loss.backward()
            optimizer.step()

    elif args.k_pairs == 3:
        for normal, p_data, n_data, p_data1, n_data1, p_data2, n_data2 in zip(train_loader, train_positives_loader[0], train_negetives_loader[0], train_positives_loader[1], train_negetives_loader[1], train_positives_loader[2], train_negetives_loader[2]):
            imgs, labels = normal
            p_imgs, _ = p_data
            n_imgs, _ = n_data
            p_imgs1, _ = p_data1
            n_imgs1, _ = n_data1
            p_imgs2, _ = p_data2
            n_imgs2, _ = n_data2
            
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            p_imgs = p_imgs.to(args.device)
            n_imgs = n_imgs.to(args.device)
            p_imgs1 = p_imgs1.to(args.device)
            n_imgs1 = n_imgs1.to(args.device)
            p_imgs2 = p_imgs2.to(args.device)
            n_imgs2 = n_imgs2.to(args.device)
        
        
            optimizer.zero_grad()
            _, normal_features = model(imgs, True)
            _, p_features = model(p_imgs, True)
            _, n_features = model(n_imgs, True)
            _, p_features1 = model(p_imgs1, True)
            _, n_features1 = model(n_imgs1, True)
            _, p_features2 = model(p_imgs2, True)
            _, n_features2 = model(n_imgs2, True)

            normal_features = normal_features[-1]
            p_features = p_features[-1]
            n_features = n_features[-1]
            p_features1 = p_features1[-1]
            n_features1 = n_features1[-1]
            p_features2 = p_features2[-1]
            n_features2 = n_features2[-1]

            loss_contrastive = torch.tensor(0.0, requires_grad=True)

            for norm_f, p_f, n_f, p_f1, n_f1, p_f2, n_f2  in zip(normal_features, p_features, n_features, p_features1, n_features1, p_features2, n_features2):
                p_fs = torch.stack([p_f, p_f1, p_f2], dim=0)
                n_fs = torch.stack([n_f, n_f1, n_f2], dim=0)
                l_c, sim_p, sim_n = contrastive(norm_f, p_fs, n_fs, temperature=args.temperature)
                loss_contrastive = loss_contrastive + l_c
                sim_ps.append(sim_p.detach().cpu())
                sim_ns.append(sim_n.detach().cpu())
            
        
            loss = loss_contrastive / len(normal_features)
            
            epoch_loss['loss'].append(loss.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            train_global_iter += 1
            writer.add_scalar("Train/loss", loss.item(), train_global_iter)
            writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)

            loss.backward()
            optimizer.step()
    
    if args.k_pairs == 1:
        avg_sim_ps = torch.mean(torch.tensor(sim_ps), dim=0).detach().cpu().numpy()
        avg_sim_ns = torch.mean(torch.tensor(sim_ns), dim=0).detach().cpu().numpy()
        writer.add_scalar("AVG_Train/sim_p", avg_sim_ps, train_global_iter)
        writer.add_scalar("AVG_Train/sim_n", avg_sim_ns, train_global_iter)
    else:

        avg_sim_ps = torch.mean(torch.cat(sim_ps, dim=0), dim=0).detach().cpu().numpy()
        avg_sim_ns = torch.mean(torch.cat(sim_ns, dim=0), dim=0).detach().cpu().numpy()
        writer.add_scalar("AVG_Train/sim_p", avg_sim_ps, train_global_iter)
        writer.add_scalar("AVG_Train/sim_n", avg_sim_ns, train_global_iter)
    # if args.tsne:
    #     tsne(net, args)
    return train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns


def load_model(args):

    if args.linear:
        model = ResNet18(args.num_classes, args.linear)
    else:
        model = ResNet18(args.num_classes)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                momentum=args.momentum,weight_decay=args.decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                                    weight_decay=args.decay)
    else:
        raise NotImplemented("Not implemented optimizer!")

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return model, criterion, optimizer, scheduler


def create_path(args):
    if args.model_path is not None:
        save_path = args.save_path
        model_save_path = save_path + 'models/'
    else:
        addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
        save_path = f'./run/exp' + f'_{args.dataset}' + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + \
        f'_{args.optimizer}' + f'_epochs_{args.epochs}' + f'_one_class_idx_{args.one_class_idx}' + \
            f'_temprature_{args.temperature}' + f'_shift_normal_{str(args.shift_normal)}' + \
            f'_preprocessing_{args.preprocessing}' + f'_seed_{args.seed}' + f'_linear_layer_{args.linear}' + f'_k_pairs_{args.k_pairs}' + '/'
        model_save_path = save_path + 'models/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        else:
            save_path = f'./run/exp-' + addr + f'_{args.dataset}' + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + \
            f'_{args.optimizer}' + f'_epochs_{args.epochs}' + f'_one_class_idx_{args.one_class_idx}' + \
                f'_temprature_{args.temperature}' + f'_shift_normal_{str(args.shift_normal)}' +\
                      f'_preprocessing_{args.preprocessing}' + f'_seed_{args.seed}' + f'_linear_layer_{args.linear}' + f'_k_pairs_{args.k_pairs}' + '/'
            model_save_path = save_path + 'models/'
            os.makedirs(model_save_path, exist_ok=True)
    
    return model_save_path, save_path


def loading_datasets(args):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        train_loader, test_loader = load_cifar10(data_path, 
                                            batch_size=args.batch_size,
                                            one_class_idx=args.one_class_idx)
    elif args.dataset == 'svhn':
        args.num_classes = 10
        train_loader, test_loader = load_svhn(data_path, 
                                            batch_size=args.batch_size,
                                            one_class_idx=args.one_class_idx)
    elif args.dataset == 'cifar100':
        args.num_classes = 20
        train_loader, test_loader = load_cifar100(data_path, 
                                                batch_size=args.batch_size,
                                                one_class_idx=args.one_class_idx)
    elif args.dataset == 'imagenet30':
        args.num_classes = 30
        train_loader, test_loader = load_imagenet30(imagenet30_path, 
                                                batch_size=args.batch_size,
                                                one_class_idx=args.one_class_idx)

    print("Start Loading noises")
    train_positives_loader, train_negetives_loader, test_positives_loader, test_negetives_loader = \
        noise_loader(args, batch_size=args.batch_size, one_class_idx=args.one_class_idx,
                    dataset=args.dataset, preprocessing=args.preprocessing, k_pairs=args.k_pairs)
    print("Loading noises finished!")

    return train_loader, test_loader, train_positives_loader, train_negetives_loader, test_positives_loader, test_negetives_loader


with open('config.json') as config_file:
    config = json.load(config_file)

root_path = config['root_path']
data_path = config['data_path']
imagenet30_path = config['imagenet30_path']
args = parsing()
args.config = config

os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
args.device=f'cuda:{args.gpu}'

# Setting seeds for reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

train_loader, test_loader, train_positives_loader, train_negetives_loader,\
      test_positives_loader, test_negetives_loader = loading_datasets(args)

model, criterion, optimizer, scheduler = load_model(args)

model_save_path, save_path = create_path(args)
writer = SummaryWriter(save_path)
args.save_path = save_path
train_global_iter = 0

args.last_lr = args.learning_rate
for epoch in range(0, args.epochs):
    print('epoch', epoch + 1, '/', args.epochs)
    args.e_holder = str(epoch)
    train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns =\
          train_one_class(train_loader, train_positives_loader, train_negetives_loader, \
                          model, train_global_iter, optimizer, args.device, writer)
    
    writer.add_scalar("AVG_Train/avg_loss", np.mean(epoch_loss['loss']), epoch)
    writer.add_scalar("AVG_Train/avg_loss_contrastive", np.mean(epoch_loss['contrastive']), epoch)
    writer.add_scalar("Train/lr", args.last_lr, epoch)
    print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])}")
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path,f'model_params_epoch_{epoch}.pt'))

    args.last_lr = scheduler.get_last_lr()[0]
    last_sim_ps = avg_sim_ps
    last_sim_ns = avg_sim_ns

torch.save(model.state_dict(), os.path.join(save_path,'last_params.pt'))
