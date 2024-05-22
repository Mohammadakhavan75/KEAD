import os
import torch
import argparse
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from contrastive import cosine_similarity
from dataset_loader import load_np_dataset
from torchvision.transforms import transforms
from dataset_loader import SVHN, get_subclass_dataset
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
    
    parser.add_argument('--dataset', default='cifar10', type=str, help='run index')
    parser.add_argument('--run_index', default=0, type=int, help='run index')
    parser.add_argument('--one_class_idx', default=None, type=int, help='run index')
    parser.add_argument('--auc_cal', default=0, type=int, help='run index')
    parser.add_argument('--noise', default=None, type=str, help='noise')
    parser.add_argument('--csv', action="store_true", help='noise')
    parser.add_argument('--score', default='mahal', type=str, help='noise')
    parser.add_argument('--thresh', default=0, type=int, help='noise')
    parser.add_argument('--svm_model', default=None, help='noise')
    parser.add_argument('--aug', default=None, type=str, help='noise')
    
    args = parser.parse_args()

    return args


def eval_auc_one_class(eval_in, eval_out, mahal_thresh, net, args):

    # print("Evaluating...")
    net = net.to(args.device)
    net.eval()  # enter train mode

    pred_in = []
    pred_out = []
    targets_in_list = []
    targets_out_list = []
    in_features_list = []
    out_features_list = []
    with torch.no_grad():
        for data_in, data_out in zip(eval_in, eval_out):       
            inputs_in, targets_in = data_in
            inputs_out, targets_out = data_out
            if args.dataset == 'anomaly':
                targets_in_list.extend(targets_in)
                targets_out_list.extend(targets_out)
            else:
                targets_in_list.extend([1 for _ in range(len(targets_in))])
                targets_out_list.extend([0 for _ in range(len(targets_out))])
            inputs_in , inputs_out = inputs_in.to(args.device) , inputs_out.to(args.device)

            preds, normal_features = net(inputs_in, True)     
            preds, out_features = net(inputs_out, True)     
            
            if 'CSI' in args.model_path:
                in_features_list.append(normal_features['penultimate'])
                out_features_list.append(out_features['penultimate'])
            else:
                in_features_list.append(normal_features[-1])
                out_features_list.append(out_features[-1])
    
        in_features_list = torch.cat(in_features_list, dim=0)
        out_features_list = torch.cat(out_features_list, dim=0)

        if args.score == 'mahal':
            for in_feature in in_features_list:
                pred_in.append(mahalanobis_distance(in_feature, mean, inv_covariance) < mahal_thresh)
            for out_feature in out_features_list:
                pred_out.append(not (mahalanobis_distance(out_feature, mean, inv_covariance) > mahal_thresh))
            
            targets = torch.cat([torch.tensor(targets_in_list), torch.tensor(targets_out_list)], dim=0)
            preds = torch.cat([torch.tensor(pred_in), torch.tensor(pred_out)], dim=0)
            auc = roc_auc_score(to_np(targets), to_np(preds))
        elif args.score == 'svm':
            if args.dataset == 'anomaly':
                targets = torch.cat([torch.tensor(targets_in_list), torch.tensor(targets_out_list)], dim=0)
                f_list = torch.cat([in_features_list, out_features_list], dim=0)
                preds = torch.tensor(args.svm_model.predict(f_list.detach().cpu().numpy()))
                preds = (preds + 1)/2
                pp = []
                for k in range(len(preds)):
                    if preds[k] == targets[k]:
                        pp.append(1)
                    else:
                        pp.append(0)
                pp = torch.tensor(pp)
                auc = roc_auc_score(to_np(targets), to_np(pp))      

            else:
                # for in_feature in in_features_list:
                pred_in.append(torch.tensor(args.svm_model.predict(in_features_list.detach().cpu().numpy())==1))
                # for out_feature in out_features_list:
                pred_out.append(torch.tensor(args.svm_model.predict(out_features_list.detach().cpu().numpy())==-1))
                pred_in = np.concatenate(pred_in)
                pred_out = np.concatenate(pred_out)
                targets = torch.cat([torch.tensor(targets_in_list), torch.tensor(targets_out_list)], dim=0)
                preds = torch.cat([torch.tensor(pred_in), torch.tensor(pred_out)], dim=0)
                preds = (preds + 1)/2
                auc = roc_auc_score(to_np(targets), to_np(preds))

    return auc

def eval_auc_anomaly(loader, net, args):
    net = net.to(args.device)
    net.eval()  # enter train mode

    preds = []
    targets_list = []
    anomaly_list = []
    features_list = []
    with torch.no_grad():
        for inputs, targets, anomaly in loader:
            targets_list.extend(targets)
            anomaly_list.extend(anomaly)
            inputs = inputs.to(args.device)

            _, normal_features = net(inputs, True)     
            
            if 'CSI' in args.model_path:
                features_list.append(normal_features['penultimate'])
            else:
                features_list.append(normal_features[-1])
    
        features_list = torch.cat(features_list, dim=0)

        preds = torch.tensor(args.svm_model.predict(features_list.detach().cpu().numpy()))
        preds = (preds + 1)/2
        pp = []
        for k in range(len(preds)):
            if preds[k] == anomaly_list[k]:
                pp.append(1)
            else:
                pp.append(0)
        pp = torch.tensor(pp)
        anomaly_list = torch.tensor(anomaly_list)
        auc = roc_auc_score(to_np(anomaly_list), to_np(pp))      

    return auc


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


def noise_loading(dataset, noise):
    if dataset == 'cifar10':
        np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/labels_test.npy'
        np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/'
    elif dataset == 'svhn':
        np_test_target_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Test_AC/labels_test.npy'
        np_test_root_path = '/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/SVHN_Test_AC/'

    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor()])

    np_test_img_path = np_test_root_path + noise + '.npy'
    test_dataset_noise = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, train=False)

    return test_dataset_noise


def feature_extraction(loader, net):
    print("extracting features...")
    net = net.to(args.device)
    net.eval()  # enter train mode
    
    features = []
    with torch.no_grad():
        for data_in in loader:
            inputs_in, _ = data_in
            
            inputs = inputs_in.to(args.device)

            _, normal_features = net(inputs, True)            
            # features.append(normal_features['penultimate'])
            if 'CSI' in args.model_path:
                features.append(normal_features['penultimate'])
            else:
                features.append(normal_features[-1])
    
    features = torch.cat(features, dim=0)
    return features

os.environ['CUDA_VISIBLE_DEVICES']='1'

args = parsing()
if 'CSI' in args.model_path:
    from cifar10.models.resnet import ResNet18, ResNet34, ResNet50
else:
    from cifar10.model import ResNet18, ResNet34, ResNet50
torch.manual_seed(args.seed)
model, criterion, optimizer, scheduler = load_model(args)

root_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
train_global_iter = 0
global_eval_iter = 0
best_acc = 0.0
best_loss = np.inf
args.last_lr = args.learning_rate


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data = torchvision.datasets.CIFAR10(
        root_path, train=True, transform=test_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(
        root_path, train=False, transform=test_transform, download=True)
elif args.dataset == 'svhn':
    train_data = SVHN(root=root_path, split="train", transform=test_transform)
    test_data = SVHN(root=root_path, split="test", transform=test_transform)
elif args.dataset == 'cifar100':
    train_data = torchvision.datasets.CIFAR100(
        root_path, train=True, transform=test_transform, download=True)
    test_data = torchvision.datasets.CIFAR100(
        root_path, train=False, transform=test_transform, download=True)
elif args.dataset == 'custom':
    np_test_img_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_labels.npy'
    train_data = torchvision.datasets.CIFAR10(
        root_path, train=True, transform=test_transform, download=True)
    test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)
elif args.dataset == 'custom6':
    np_test_img_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_s6.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_labels_s6.npy'
    train_data = torchvision.datasets.CIFAR10(
        root_path, train=True, transform=test_transform, download=True)
    test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)
elif args.dataset == 'anomaly':

    np_test_img_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_labels.npy'
    
    train_data = torchvision.datasets.CIFAR10(
        root_path, train=True, transform=test_transform, download=True)
    test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='anomaly', train=False)
elif args.dataset == 'aug':
    if args.aug:
        np_test_img_path = f'/storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR10_Test_AC/{args.aug}.npy'
        np_test_target_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_labels.npy'
    else:
        np_test_img_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test.npy'
        np_test_target_path = '/storage/users/makhavan/CSI/exp10/new_contrastive/new_test_set/cifar10_test_labels.npy'
    
    train_data = torchvision.datasets.CIFAR10(
        root_path, train=True, transform=test_transform, download=True)
    test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

idxes = [i for i in range(10)]
idxes.pop(args.one_class_idx)
eval_in_train_data = get_subclass_dataset(train_data, args.one_class_idx)
eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)

eval_in_train = DataLoader(eval_in_train_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
eval_in = DataLoader(eval_in_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

if args.score == 'mahal':
    mean, inv_covariance, in_features = mahalanobis_parameters(eval_in_train, model, args)
if args.score == 'svm':
    from sklearn.svm import OneClassSVM
    # svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    svm_model = OneClassSVM(kernel='sigmoid', gamma='auto', nu=0.1)
    features_in = feature_extraction(eval_in_train, model)
    print(f"Fitting svm model {features_in.shape}")
    svm_model.fit(features_in.detach().cpu().numpy())
    args.svm_model = svm_model
if args.score == 'cosine':
    pass

if args.score == 'mahal':
    in_distance = evaluator(model, eval_in, mean, inv_covariance, args)
    print(f"Original distance: {in_distance}")
else:
    in_distance = torch.tensor(0)


if args.noise:
    test_dataset_noise = noise_loading(args.dataset, args.noise)
    
resutls = {}
aucs = []

# resutls['OD']=in_distance.detach().cpu().numpy()
for id in range(10):
    if args.noise:
        eval_out_data = get_subclass_dataset(test_dataset_noise, id)
    else:
        eval_out_data = get_subclass_dataset(test_data, id)
    eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    if args.score == 'mahal':
        m_distance = evaluator(model, eval_out, mean, inv_covariance, args)

    if args.dataset == 'anomaly':
        auc = eval_auc_anomaly(eval_out, model, args)
    else:
        auc = eval_auc_one_class(eval_in, eval_out, in_distance + args.thresh, model, args)
    aucs.append(auc)
    if args.score == 'mahal':
        resutls[id]=(m_distance - in_distance).detach().cpu().numpy()
        print(f"Evaluation distance on class {id}: {m_distance}, diff: {m_distance - in_distance}, auc: {auc}")
    elif args.score == 'svm':
        print(f"Evaluation distance on class {id}: auc: {auc}")

print(f"Average auc is: {np.mean(aucs)}")
print(f"Results of model: {args.model_path}")
print(f"In class is {args.one_class_idx}")
resutls['avg_auc']=np.mean(aucs)
if args.csv:
    import pandas as pd
    df = pd.DataFrame(resutls, index=[0])

    # Save DataFrame as CSV file
    save_path = f'./csv_results/report_class_{args.one_class_idx}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    df.to_csv(save_path+f"{args.noise}_{args.model_path.split('/')[-2]}.csv", index=False) 
