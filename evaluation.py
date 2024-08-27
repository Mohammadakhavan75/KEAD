import os
import torch
import faiss
import argparse
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from dataset_loader import load_np_dataset, load_mvtec_ad, MVTecADDataset
from torchvision.transforms import transforms
from dataset_loader import SVHN, get_subclass_dataset, sparse2coarse
from sklearn.metrics import roc_auc_score
from models.resnet import ResNet18, ResNet34, ResNet50


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


def to_np(x):
    return x.data.cpu().numpy()


def parsing():
    parser = argparse.ArgumentParser(description='Selecting parameters',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
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
                        default='sgd', help='optimizer.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float,
                        default=5, help='The update rate for learning rate.')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.9, help='The gamma param for updating learning rate.')
    parser.add_argument('--last_lr', type=float,
                        default=0, help='last learning rate.')
                        
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
    parser.add_argument('--gpu', default='1', type=str, help='noise')
    parser.add_argument('--linear', action="store_true", help='noise')
    
    args = parser.parse_args()

    return args

def knn_score(train_set, test_set, n_neighbours=1):
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def mahalanobis_parameters(loader_in, net, args):
    print("Calculating mahalanobis parameters...")
    features = torch.cat(features, dim=0)
    mean = torch.mean(features, dim=0)

    # for feature in features:
    #     feature[feature==0] += 1e-12 # To preventing zero for torch.linalg.inv
    cov = torch.cov(features.t())
    cov += 1e-12 * torch.eye(cov.shape[0]).to(args.device)
    inv_covariance = torch.linalg.inv(cov)

    return mean, inv_covariance, features


def mahalanobis_distance(x, mu, inv_cov):
    centered_x = x - mu
    mahalanobis_dist = torch.sqrt(torch.clamp(torch.matmul(torch.matmul(centered_x.unsqueeze(0), inv_cov), centered_x), min=0))

    return mahalanobis_dist


def load_data(root_path, args):
    test_transform = transforms.Compose([transforms.ToTensor()])

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
    
    elif args.dataset == 'anomaly':

        np_test_img_path = 'path_to_new_anomaly_data'
        np_test_target_path = 'path_to_new_anomaly_labels'
        
        train_data = torchvision.datasets.CIFAR10(
            root_path, train=True, transform=test_transform, download=True)
        test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='anomaly', train=False)


    elif args.dataset == 'anomaly100':

        np_test_img_path = 'path_to_new_anomaly_data'
        np_test_target_path = 'path_to_new_anomaly_label'
        anomaly_path = 'path_to_new_anomaly_config'
        
        train_data = torchvision.datasets.CIFAR100(
            root_path, train=True, transform=test_transform, download=True)
        test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='anomaly', train=False, anomaly_path=anomaly_path)
    
    elif args.dataset == 'anomalysvhn':

        np_test_img_path = 'path_to_new_anomaly_data'
        np_test_target_path = 'path_to_new_anomaly_label'
        anomaly_path = 'path_to_new_anomaly_config'
        
        train_data = SVHN(root=root_path, split="train", transform=test_transform)
        test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='anomaly', train=False, anomaly_path=anomaly_path)


    elif args.dataset == 'aug':
        if args.aug:
            np_test_img_path = f'/generalization_repo_dataset/CIFAR10_Test_AC_6/{args.aug}.npy'
            np_test_target_path = '/generalization_repo_dataset/CIFAR10_Test_AC_6/labels_test.npy'
        else:
            np_test_img_path = '/cifar10_test.npy'
            np_test_target_path = '/cifar10_test_labels.npy'
        
        train_data = torchvision.datasets.CIFAR10(
            root_path, train=True, transform=test_transform, download=True)
        test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

    elif args.dataset == 'svhn_aug':
        np_test_img_path = f'/generalization_repo_dataset/SVHN_Test_AC_6/{args.aug}.npy'
        np_test_target_path = '/generalization_repo_dataset/SVHN_Test_AC_6/labels_test.npy'
        train_data = SVHN(root=root_path, split="train", transform=test_transform)
        test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

    elif args.dataset == 'aug100':
        np_test_img_path = f'/generalization_repo_dataset/CIFAR100_Test_AC_6/{args.aug}.npy'
        np_test_target_path = '/generalization_repo_dataset/CIFAR100_Test_AC_6/labels_test.npy'
        train_data = torchvision.datasets.CIFAR100(
            root_path, train=True, transform=test_transform, download=True)
        test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

    elif args.dataset == 'mvtec_ad':
        import math
        resize=32
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(math.ceil(resize*1.14)),
            torchvision.transforms.CenterCrop(resize),
            torchvision.transforms.ToTensor()])    
        categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        train_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='train')
        test_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='test')

    # Create sub classes
    if args.one_class_idx != None:
        if (args.dataset == 'cifar100' or args.dataset == 'aug100' or args.dataset == 'anomaly100'):
            train_data.targets = sparse2coarse(train_data.targets)
            test_data.targets = sparse2coarse(test_data.targets)
            eval_in_train_data = get_subclass_dataset(train_data, args.one_class_idx)
            eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)
        else:
            eval_in_train_data = get_subclass_dataset(train_data, args.one_class_idx)
            eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)
    
    return eval_in_train_data, eval_in_data, train_data, test_data



def eval_auc_one_class(eval_in, eval_out, train_features_in, mahal_thresh, net, args):

    net = net.to(args.device)
    net.eval()  # enter valid mode

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

            preds_in, normal_features = net(inputs_in, True)     
            preds_out, out_features = net(inputs_out, True)     
            
            in_features_list.append(normal_features[-1])
            out_features_list.append(out_features[-1])
        
        in_features_list = torch.cat(in_features_list, dim=0)
        out_features_list = torch.cat(out_features_list, dim=0)

        if args.score == 'mahalanobis':
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
                pred_in.append(torch.tensor(args.svm_model.predict(in_features_list.detach().cpu().numpy())==1))
                pred_out.append(torch.tensor(args.svm_model.predict(out_features_list.detach().cpu().numpy())==-1))
                pred_in = np.concatenate(pred_in)
                pred_out = np.concatenate(pred_out)
                targets = torch.cat([torch.tensor(targets_in_list), torch.tensor(targets_out_list)], dim=0)
                preds = torch.cat([torch.tensor(pred_in), torch.tensor(pred_out)], dim=0)
                preds = (preds + 1)/2
                auc = roc_auc_score(to_np(targets), to_np(preds))
        elif args.score == 'knn':
            f_list = torch.cat([in_features_list, out_features_list], dim=0)
            distances = knn_score(to_np(train_features_in), to_np(f_list))
            targets = torch.cat([torch.tensor(targets_in_list), torch.tensor(targets_out_list)], dim=0)
            auc = roc_auc_score(targets, distances)
    return auc


def eval_auc_anomaly(loader, train_features_in, net, args):
    net = net.to(args.device)
    net.eval()  # enter valid mode

    preds = []
    anomaly_list = []
    features_list = []
    with torch.no_grad():
        for inputs, _, anomaly in loader:

            anomaly_list.extend(anomaly)
            inputs = inputs.to(args.device)

            _, normal_features = net(inputs, True)     
            features_list.append(normal_features[-1])
    
        features_list = torch.cat(features_list, dim=0)
        if args.score == 'svm':
            preds = torch.tensor(args.svm_model.predict(features_list.detach().cpu().numpy()))
            preds = (preds + 1)/2
            
            anomaly_list = [1-x for x in anomaly_list]
            auc = roc_auc_score(anomaly_list, preds)      
        elif args.score == 'knn':
            distances = knn_score(to_np(train_features_in), to_np(features_list))
            auc = roc_auc_score(anomaly_list, distances)

    return auc


def load_model(args):

    if args.linear or 'linear_layer_True' in args.model_path:
        model = ResNet18(10, True)
    else:
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
        m = torch.load(args.model_path)
        model.load_state_dict(m)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return model, criterion, optimizer, scheduler


def noise_loading(dataset, noise):
    if dataset == 'cifar10':
        np_test_target_path = 'path_to_genralization_labels'
        np_test_root_path = 'path_to_genralization_folder'
    elif dataset == 'svhn':
        np_test_target_path = 'path_to_genralization_labels'
        np_test_root_path = 'path_to_genralization_folder'


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
            features.append(normal_features[-1])
    
    features = torch.cat(features, dim=0)
    return features



args = parsing()
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
torch.manual_seed(args.seed)
model, criterion, optimizer, scheduler = load_model(args)

root_path = 'D:/Datasets/data/'
args.last_lr = args.learning_rate
in_distance = torch.tensor(0)

eval_in_train_data, eval_in_data, train_data, test_data = load_data(root_path, args)

eval_in_train = DataLoader(eval_in_train_data, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
eval_in = DataLoader(eval_in_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

if args.score == 'svm':
    from sklearn.svm import OneClassSVM
    svm_model = OneClassSVM(kernel='sigmoid', gamma='auto', nu=0.2)
    train_features_in = feature_extraction(eval_in_train, model)
    print(f"Fitting svm model {train_features_in.shape}")
    svm_model.fit(train_features_in.detach().cpu().numpy())
    args.svm_model = svm_model
if args.score == 'knn':
    train_features_in = feature_extraction(eval_in_train, model)

if args.score == 'mahalanobis':
    train_features_in = feature_extraction(eval_in_train, model)
    mean, inv_covariance, features = mahalanobis_parameters(train_features_in, args)

if args.noise:
    test_dataset_noise = noise_loading(args.dataset, args.noise)


resutls = {}
aucs = []

# resutls['OD']=in_distance.detach().cpu().numpy()
if args.dataset == 'cifar100':
    classes=20
else:
    classes=10
if args.dataset == 'anomaly' or args.dataset == 'anomaly100' or args.dataset == 'anomalysvhn':
    auc = eval_auc_anomaly(eval_in, train_features_in, model, args)
    print(f"Anomaly auc is: {auc}")
    exit()
if args.one_class_idx != None:
    for id in range(classes):
        if id == args.one_class_idx:
            continue
        
        if args.noise:
            eval_out_data = get_subclass_dataset(test_dataset_noise, id)
        else:
            eval_out_data = get_subclass_dataset(test_data, id)
        eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

        if args.dataset == 'anomaly':
            auc = eval_auc_anomaly(eval_out, train_features_in, model, args)
        else:
            auc = eval_auc_one_class(eval_in, eval_out, train_features_in, in_distance + args.thresh, model, args)
        aucs.append(auc)
        
        print(f"Evaluation distance on class {id}: auc: {auc}")
else:
    if args.noise:
        eval_out_data = test_dataset_noise
    else:
        eval_out_data = test_data

    eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    if args.dataset == 'anomaly':
        auc = eval_auc_anomaly(eval_out, train_features_in, model, args)
    else:
        auc = eval_auc_one_class(eval_in, eval_out, train_features_in, in_distance + args.thresh, model, args)

    aucs.append(auc)
    
    print(f"Evaluation distance on class {id}: auc: {auc}")


print(f"Average auc is: {np.mean(aucs)}")
print(f"Results of model: {args.model_path}")
print(f"In class is {args.one_class_idx}")
resutls['avg_auc']=np.mean(aucs)
if args.csv:
    import pandas as pd
    df = pd.DataFrame(resutls, index=[0])

    # Save DataFrame as CSV file
    ev = args.model_path.split('_')[1]
    save_path = f'./csv_results/{ev}/dataset_{args.dataset}/report_class_{args.one_class_idx}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    df.to_csv(save_path+f"{args.model_path.split('/')[-2]}.csv", index=False) 
