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


from dataset_loader import noise_loader, load_cifar10

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
                        
    parser.add_argument('--lamb', type=float, default=1, help='loss scaling parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')
    

    parser.add_argument('--run_index', default=0, type=int, help='run index')
    parser.add_argument('--one_class_idx', default=None, type=int, help='run index')
    parser.add_argument('--auc_cal', default=0, type=int, help='run index')
    
    args = parser.parse_args()

    return args



def eval_auc_one_class(eval_in, eval_out, net, global_eval_iter, criterion, device):

    print("Evaluating...")
    net = net.to(device)
    net.eval()  # enter train mode

    # track train classification accuracy
    epoch_accuracies = {
        'acc': [],
        'auc': [],
    }
    epoch_loss = {
        'loss': [],
    }
    with torch.no_grad():
        for data_in, data_out in zip(eval_in, eval_out):
            
            inputs_in, targets_in = data_in
            targets_in_auc = [1 for _ in targets_in]
            targets_in_auc = torch.tensor(targets_in_auc, dtype=torch.float32)

            inputs_out, targets_out = data_out
            targets_out_auc = [0 for _ in targets_out]
            targets_out_auc = torch.tensor(targets_out_auc, dtype=torch.float32)

            
            inputs_ = torch.cat((inputs_in, inputs_out), dim=0)
            targets_auc_ = torch.cat((targets_in_auc, targets_out_auc), dim=0)
            targets_ = torch.cat((targets_in, targets_out), dim=0)
            random_indices = torch.randperm(inputs_.size(0))
            inputs = inputs_[random_indices]
            targets = targets_[random_indices]
            targets_auc = targets_auc_[random_indices]

            inputs , targets = inputs.to(device) , targets.to(device)

            preds, normal_features = net(inputs, True)            

            loss = criterion(preds, targets)
            
            acc = accuracy_score(list(to_np(torch.argmax(torch.softmax(preds, dim=1), axis=1))), list(to_np(targets)))
            auc = roc_auc_score(to_np(torch.argmax(torch.softmax(preds, dim=1), axis=1) == targets) , to_np(targets_auc))

            # Logging section
            epoch_accuracies['acc'].append(acc)
            epoch_accuracies['auc'].append(auc)
            epoch_loss['loss'].append(loss.item())

            global_eval_iter += 1

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

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)



    return model, criterion, optimizer, scheduler


args = parsing()
torch.manual_seed(args.seed)
model, criterion, optimizer, scheduler = load_model(args)

cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'


train_global_iter = 0
global_eval_iter = 0
best_acc = 0.0
best_loss = np.inf
args.last_lr = args.learning_rate

from dataset_loader import get_subclass_dataset
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

test_data = torchvision.datasets.CIFAR10(
    cifar10_path, train=False, transform=test_transform, download=True)

idxes = [i for i in range(10)]
idxes.pop(args.one_class_idx)
eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)
eval_out_data = get_subclass_dataset(test_data, idxes)

eval_in = DataLoader(eval_in_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

global_eval_iter, eval_loss, eval_acc = eval_auc_one_class(eval_in, eval_out, model, global_eval_iter, criterion, args.device)

print(f"Evaluation/avg_loss: {np.mean(eval_loss['loss'])}, Evaluation/avg_acc: {np.mean(eval_acc['acc'])}, Evaluation/avg_auc: {np.mean(eval_acc['auc'])}")
