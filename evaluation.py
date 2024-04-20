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
# from cifar10.models.resnet import ResNet18, ResNet34, ResNet50
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score
from contrastive import cosine_similarity

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
    parser.add_argument('--noise', default='snow', type=str, help='noise')
    parser.add_argument('--csv', action="store_true", help='noise')
    parser.add_argument('--score', default='mahal', type=str, help='noise')
    
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
            auc = roc_auc_score(to_np(targets_auc), to_np(torch.argmax(torch.softmax(preds, dim=1), axis=1) == args.one_class_idx))

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


def mahalanobis_parameters(loader_in, net, args):
    print("Calculating mahalanobis parameters...")
    net = net.to(args.device)
    net.eval()  # enter train mode
    
    features = []
    with torch.no_grad():
        for data_in in loader_in:
            inputs_in, _ = data_in
            
            inputs = inputs_in.to(args.device)

            _, normal_features = net(inputs, True)            
            # features.append(normal_features['penultimate'])
            features.append(normal_features[-1])
    
    features = torch.cat(features, dim=0)
    mean = torch.mean(features, dim=0)
    
    # for feature in features:
    #     feature[feature==0] += 1e-12 # To preventing zero for torch.linalg.inv
    cov = torch.cov(features.t())
    cov += 1e-12 * torch.eye(cov.shape[0]).to(args.device)
    # inv_covariance = torch.linalg.inv(torch.cov(features.t()))
    inv_covariance = torch.linalg.inv(cov)

    return mean, inv_covariance, features


def mahalanobis_distance(x, mu, inv_cov):

  centered_x = x - mu

  mahalanobis_dist = torch.sqrt(torch.clamp(torch.matmul(torch.matmul(centered_x.unsqueeze(0), inv_cov), centered_x), min=0))

  return mahalanobis_dist


def out_features_extractor(eval_out, model, args):
    out_features = []
    model = model.to(args.device)
    for data_in in eval_out:
        inputs_in, _ = data_in

        inputs = inputs_in.to(args.device)
        
        preds, normal_features = model(inputs, True)  
        out_features.append(normal_features['penultimate'])          
        # break
    out_features = torch.cat(out_features, dim=0)
    
    return out_features


def evaluator(model, eval_out, mean, inv_covariance, args):
    out_features = []
    model = model.to(args.device)
    for data_in in eval_out:
        inputs_in, _ = data_in

        inputs = inputs_in.to(args.device)
        
        preds, normal_features = model(inputs, True)  
        out_features.append(normal_features[-1])
        # break
    out_features = torch.cat(out_features, dim=0)
    out_dist = []
    for out_feature in out_features:
        out_dist.append(mahalanobis_distance(out_feature, mean, inv_covariance))

    return torch.mean(torch.tensor(out_dist))


def in_similarity(loader_in, net, args):
    print("Calculating mahalanobis parameters...")
    net = net.to(args.device)
    net.eval()  # enter train mode
    
    features = []
    with torch.no_grad():
        for data_in in loader_in:
            inputs_in, _ = data_in
            
            inputs = inputs_in.to(args.device)

            _, normal_features = net(inputs, True)            
            # features.append(normal_features['penultimate'])
            features.append(normal_features[-1])
    
    features = torch.cat(features, dim=0)
    similarity = torch.zeros((features.shape[0], features.shape[0])).to(args.device)
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            similarity[i,j] = cosine_similarity(f1, f2)
    
    return similarity


from dataset_loader import load_np_dataset
def noise_loading(noise):
    np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/data_aug/CorCIFAR10_test/labels.npy'
    np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/'

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    np_test_img_path = np_test_root_path + noise + '.npy'
    test_dataset_noise = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, train=False)

    return test_dataset_noise



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

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

test_data = torchvision.datasets.CIFAR10(
    cifar10_path, train=False, transform=test_transform, download=True)

idxes = [i for i in range(10)]
idxes.pop(args.one_class_idx)
eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)

eval_in = DataLoader(eval_in_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

if args.score == 'mahal':
    mean, inv_covariance, in_features = mahalanobis_parameters(eval_in, model, args)
if args.score == 'cosine':
    pass

in_distance = evaluator(model, eval_in, mean, inv_covariance, args)

print(f"Original distance: {in_distance}")

test_dataset_noise = noise_loading(args.noise)
resutls = {}
for id in range(10):
    eval_out_data = get_subclass_dataset(test_dataset_noise, id)
    eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    m_distance = evaluator(model, eval_out, mean, inv_covariance, args)

    resutls[id]=(m_distance - in_distance).detach().cpu().numpy()
    print(f"Evaluation distance on class {id}: {m_distance}, diff: {m_distance - in_distance}")

if args.csv:
    import pandas as pd
    df = pd.DataFrame(resutls, index=[0])

    # Save DataFrame as CSV file
    save_path = f'./csv_results/report_class_{args.one_class_idx}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    df.to_csv(save_path+f"{args.noise}.csv", index=False) 
