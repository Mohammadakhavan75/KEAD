import os
import torch
import random
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


from dataset_loader import noise_loader, load_cifar10, load_svhn, load_cifar100, csi_aug

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
                        default=5, help='The update rate for learning rate.')
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 50, 200, 500],
                         help='A list of milestones')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.9, help='The gamma param for updating learning rate.')
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
    parser.add_argument('--multi_neg', action="store_true", help='A list of milestones')
    args = parser.parse_args()

    return args


def train_one_class(train_loader, positives, negetives, shuffle_loader,
                     net, train_global_iter, optimizer, device, writer):

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
    
    if args.shift_normal:
        a_ = list(shuffle_loader)
        b_ = []
        for aa_ in a_:
            b_.append(aa_[0])
        shuffle_imgs_list = torch.concatenate(b_)

    # for normal, ps, ns, sl in zip(train_loader, positives, negetives, shuffle_loader):
    for normal, ps, ns in zip(train_loader, positives, negetives):
    # for normal in train_loader:

        imgs, labels = normal
        positive_imgs, _ = ps
        negative_imgs, _ = ns

        if args.shift_normal:
            positive_noraml_img = shuffle_imgs_list[torch.randint(0, len(shuffle_imgs_list), size=(1,))]
            positive_noraml_img =  positive_noraml_img.to(args.device)

        if args.csi_aug:
            csi_transform = csi_aug()
            imgs = csi_transform(imgs)


        imgs, labels = imgs.to(args.device), labels.to(args.device)
        positive_imgs, negative_imgs = positive_imgs.to(args.device), negative_imgs.to(args.device)

        optimizer.zero_grad()
        _, normal_features = model(imgs, True)
        _, positive_features = model(positive_imgs, True)
        _, negative_features = model(negative_imgs, True)
        if args.shift_normal:
            _, positive_noraml_features = model(positive_noraml_img, True)

        normal_features = normal_features[-1]
        positive_features = positive_features[-1]
        negative_features = negative_features[-1]
        if args.shift_normal:
            positive_noraml_features = positive_noraml_features[-1]
        
        # Calculate loss contrastive for layer
        loss_contrastive = 0
        if args.shift_normal:
            if len(negative_features) == len(normal_features):
                for norm_f, pos_f, neg_f, pn_f in zip(normal_features, positive_features, negative_features, positive_noraml_features):
                    pos_sl_f = torch.stack([pos_f, pn_f])
                    loss_contrastive = loss_contrastive + contrastive(
                        norm_f, pos_sl_f, neg_f, temperature=args.temperature)

        else:
            if len(negative_features) == len(normal_features):
                for norm_f, pos_f, neg_f in zip(normal_features, positive_features, negative_features):

                    pos_sl_f = pos_f
                    loss_contrastive = loss_contrastive + contrastive(
                        norm_f, pos_sl_f, neg_f, temperature=args.temperature)


        loss = loss_contrastive / len(normal_features)
        
        epoch_loss['loss'].append(loss.item())
        epoch_loss['contrastive'].append(loss_contrastive.item())

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss.item(), train_global_iter)

        loss.backward()
        optimizer.step()

    return train_global_iter, epoch_loss, epoch_accuracies 


def train_one_class_multi_neg(train_loader, positives, negetives, negetives2, shuffle_loader,
                     net, train_global_iter, optimizer, device, writer):

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
    

    a_ = list(negetives)
    b_ = []
    for aa_ in a_:
        b_.append(aa_[0])
    negative_imgs_list = torch.concatenate(b_)

    a_ = list(positives)
    b_ = []
    for aa_ in a_:
        b_.append(aa_[0])
    positives_imgs_list = torch.concatenate(b_)

    a_ = list(shuffle_loader)
    b_ = []
    for aa_ in a_:
        b_.append(aa_[0])
    shuffle_imgs_list = torch.concatenate(b_)


    for normal, ps, ns, ns2 in zip(train_loader, positives, negetives, negetives2):
    # for normal in train_loader:

        imgs, labels = normal
        positive_imgs, _ = ps
        negative_imgs, _ = ns
        negative_imgs2, _ = ns2
        # positive_noraml_img, _ = sl

        positive_noraml_img = shuffle_imgs_list[torch.randint(0, len(shuffle_imgs_list), size=(1,))]

        if args.csi_aug:
            csi_transform = csi_aug()
            imgs = csi_transform(imgs)


        imgs, labels = imgs.to(args.device), labels.to(args.device)
        positive_imgs, negative_imgs, negative_imgs2, positive_noraml_img = \
            positive_imgs.to(args.device), negative_imgs.to(args.device), negative_imgs2.to(args.device), positive_noraml_img.to(args.device)

        optimizer.zero_grad()
        _, normal_features = model(imgs, True)
        _, positive_features = model(positive_imgs, True)
        _, negative_features = model(negative_imgs, True)
        _, negative_features2 = model(negative_imgs2, True)
        if args.shift_normal:
            _, positive_noraml_features = model(positive_noraml_img, True)

        positive_features = positive_features[-1]
        normal_features = normal_features[-1]
        negative_features = negative_features[-1]
        negative_features2 = negative_features2[-1]
        if args.shift_normal:
            positive_noraml_features = positive_noraml_features[-1]
        
        # Calculate loss contrastive for layer
        loss_contrastive = 0
        # for norm_f, pos_f, neg_f, sl_f in zip(normal_features, positive_features, negative_features, positive_noraml_features):
        if args.shift_normal:
            if len(negative_features) == len(normal_features):
                for norm_f, pos_f, neg_f, neg_f2, pn_f in zip(normal_features, positive_features, negative_features, negative_features2, positive_noraml_features):
                # for norm_f in normal_features:
                    # pos_sl_f = torch.stack([pos_f, sl_f])
                    # loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, sl_f, temperature=args.temperature))
                    # pos_sl_f = torch.stack([positive_features[0], positive_noraml_features[0]])
                    pos_sl_f = torch.stack([pos_f, pn_f])
                    neg_f = torch.stack([neg_f, neg_f2])
                    loss_contrastive = loss_contrastive + torch.sum(contrastive(
                        norm_f, pos_sl_f, neg_f, temperature=args.temperature))
            else:
                for norm_f, pos_f, pn_f in zip(normal_features, positive_features, positive_noraml_features):
                    pos_sl_f = torch.stack([pos_f, pn_f])
                    loss_contrastive = loss_contrastive + torch.sum(contrastive(
                        norm_f, pos_sl_f, negative_features, temperature=args.temperature))


        else:
            if len(negative_features) == len(normal_features):
                for norm_f, pos_f, neg_f, neg_f2 in zip(normal_features, positive_features, negative_features, negative_features2):
                # for norm_f in normal_features:
                    # pos_sl_f = torch.stack([pos_f, sl_f])
                    # loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, sl_f, temperature=args.temperature))
                    # pos_sl_f = torch.stack([positive_features[0], positive_noraml_features[0]])
                    pos_sl_f = pos_f
                    neg_f = torch.stack([neg_f, neg_f2])
                    loss_contrastive = loss_contrastive + torch.sum(contrastive(
                        norm_f, pos_sl_f, neg_f, temperature=args.temperature))
            else:
                for norm_f, pos_f in zip(normal_features, positive_features):
                    pos_sl_f = pos_f
                    loss_contrastive = loss_contrastive + torch.sum(contrastive(
                        norm_f, pos_sl_f, negative_features, temperature=args.temperature))


        # loss_ce = criterion(preds, labels)
        # loss = loss_contrastive
        loss = loss_contrastive / len(normal_features)
        
        epoch_loss['loss'].append(loss.item())
        epoch_loss['contrastive'].append(loss_contrastive.item())

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss.item(), train_global_iter)

        loss.backward()
        optimizer.step()

    return train_global_iter, epoch_loss, epoch_accuracies 


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

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    
    milestones = args.milestones  # Epochs at which to decrease learning rate
    gamma = 0.1  # Factor by which to decrease learning rate at milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
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
            f'_preprocessing_{args.preprocessing}' + f'_seed_{args.seed}' + f'_csi_aug_{args.csi_aug}' + f'_multi_neg_{args.multi_neg}' + '/'
        model_save_path = save_path + 'models/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        else:
            save_path = f'./run/exp-' + addr + f'_{args.dataset}' + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + \
            f'_{args.optimizer}' + f'_epochs_{args.epochs}' + f'_one_class_idx_{args.one_class_idx}' + \
                f'_temprature_{args.temperature}' + f'_preprocessing_{args.preprocessing}' + f'_seed_{args.seed}' +\
                    f'_csi_aug_{args.csi_aug}' + f'_multi_neg_{args.multi_neg}' + '/'
            model_save_path = save_path + 'models/'
            os.makedirs(model_save_path, exist_ok=True)
    return model_save_path, save_path

args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES']= args.device_num
model, criterion, optimizer, scheduler = load_model(args)

if args.dataset == 'cifar10':
    cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    train_loader, test_loader, shuffle_loader = load_cifar10(cifar10_path, 
                                                             batch_size=args.batch_size,
                                                             one_class_idx=args.one_class_idx, 
                                                             tail_normal=args.tail_normal)
elif args.dataset == 'svhn':
    svhn_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    train_loader, test_loader, shuffle_loader = load_svhn(svhn_path, 
                                                          batch_size=args.batch_size,
                                                          one_class_idx=args.one_class_idx, 
                                                          tail_normal=args.tail_normal)
elif args.dataset == 'cifar100':
    cifar100_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
    train_loader, test_loader, shuffle_loader = load_cifar100(cifar100_path, 
                                                          batch_size=args.batch_size,
                                                          one_class_idx=args.one_class_idx)

print("Start Loading noises")
if args.multi_neg:
    train_positives, train_negetives, train_negetives2, test_positives, test_negetives = \
        noise_loader(batch_size=args.batch_size, one_class_idx=args.one_class_idx,
                    tail_negative=args.tail_negative, tail_positive=args.tail_positive,
                    dataset=args.dataset, preprocessing=args.preprocessing, multi_neg=args.multi_neg)
else:
    train_positives, train_negetives, test_positives, test_negetives = \
        noise_loader(batch_size=args.batch_size, one_class_idx=args.one_class_idx,
                    tail_negative=args.tail_negative, tail_positive=args.tail_positive,
                    dataset=args.dataset, preprocessing=args.preprocessing)
print("Loading noises finished!")

model_save_path, save_path = create_path(args)
writer = SummaryWriter(save_path)

train_global_iter = 0
args.last_lr = args.learning_rate

for epoch in range(0, args.epochs):
    print('epoch', epoch + 1, '/', args.epochs)
    if args.one_class_idx != None:
        model.linear.requires_grad_ = False
        if args.multi_neg:
            train_global_iter, epoch_loss, epoch_accuracies = train_one_class_multi_neg(train_loader, train_positives, train_negetives, train_negetives2, shuffle_loader, model, train_global_iter, optimizer, args.device, writer)
        else:
            train_global_iter, epoch_loss, epoch_accuracies = train_one_class(train_loader, train_positives, train_negetives, shuffle_loader, model, train_global_iter, optimizer, args.device, writer)
        
    writer.add_scalar("AVG_Train/avg_loss", np.mean(epoch_loss['loss']), epoch)
    writer.add_scalar("AVG_Train/avg_loss_contrastive", np.mean(epoch_loss['contrastive']), epoch)
    writer.add_scalar("Train/lr", args.last_lr, epoch)
    print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])}")
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path,f'model_params_epoch_{epoch}.pt'))

    args.last_lr = scheduler.get_last_lr()[0]

    scheduler.step()

torch.save(model.state_dict(), os.path.join(save_path,'last_params.pt'))