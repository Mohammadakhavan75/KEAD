import os
import torch
import faiss
import argparse
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from dataset_loader import load_np_dataset, load_mvtec_ad, MVTecADDataset, VisADataset
from torchvision.transforms import transforms
from dataset_loader import SVHN, get_subclass_dataset, sparse2coarse
from sklearn.metrics import roc_auc_score
from models.resnet import ResNet18, ResNet50
from models.resnet_imagenet import resnet18, resnet50
from sklearn.metrics import accuracy_score


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
    parser.add_argument('--gpu', default='0', type=str, help='noise')
    parser.add_argument('--linear', action="store_true", help='noise')
    parser.add_argument('--model', default='resnet18', type=str, help='noise')
    parser.add_argument('--img_size', default=32, type=int, help='noise')
    parser.add_argument('--num_classes', default=10, type=int, help='noise')
    
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
    if args.noise:
        print(f"Loading noise {args.noise} as test data")
        if args.dataset == 'cifar10':
            np_test_img_path = root_path + f'/generalization_repo_dataset/cifar10_Test_s5/{args.noise}.npy'
            np_test_target_path = root_path + '/generalization_repo_dataset/cifar10_Test_s5/labels.npy'
            
            train_data = torchvision.datasets.CIFAR10(
                root_path, train=True, transform=test_transform, download=True)
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

        elif args.dataset == 'svhn':
            np_test_img_path = root_path + f'/generalization_repo_dataset/svhn_Test_s5/{args.noise}.npy'
            np_test_target_path = root_path + '/generalization_repo_dataset/svhn_Test_s5/labels.npy'
            train_data = SVHN(root=root_path, split="train", transform=test_transform)
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

        elif args.dataset == 'cifar100':
            np_test_img_path = root_path + f'/generalization_repo_dataset/cifar100_Test_s5/{args.noise}.npy'
            np_test_target_path = root_path + '/generalization_repo_dataset/cifar100_Test_s5/labels.npy'
            train_data = torchvision.datasets.CIFAR100(
                root_path, train=True, transform=test_transform, download=True)
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

        elif args.dataset == 'mvtec_ad':
            import math
            resize=args.img_size
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(math.ceil(resize*1.14)),
                torchvision.transforms.CenterCrop(resize),
                torchvision.transforms.ToTensor()])    
            categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
            train_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='train')
            
            np_test_img_path = root_path + f'/generalization_repo_dataset/mvtec_ad_Test_s5/{args.noise}.npy'
            np_test_target_path = root_path + '/generalization_repo_dataset/mvtec_ad_Test_s5/labels.npy'
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

        elif args.dataset == 'visa':
            import math
            resize=args.img_size
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(math.ceil(resize*1.14)),
                torchvision.transforms.CenterCrop(resize),
                torchvision.transforms.ToTensor()])

            categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
            train_data = VisADataset(root_path, transform=transform, categories=categories, phase='normal')

            np_test_img_path = root_path + f'/generalization_repo_dataset/visa_Train_s5/{args.noise}.npy'
            np_test_target_path = root_path + '/generalization_repo_dataset/visa_Train_s5/labels.npy'
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10', train=False)

        else:
            raise NotImplementedError("Not impelemented dataset!")

    else:
        print("Loading original test data")
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
        
        elif args.dataset == 'mvtec_ad':
            import math
            resize=args.img_size
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(math.ceil(resize*1.14)),
                torchvision.transforms.CenterCrop(resize),
                torchvision.transforms.ToTensor()])    
            categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
            train_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='train')
            test_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='test')

        elif args.dataset == 'visa':
            import math
            resize=args.img_size
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(math.ceil(resize*1.14)),
                torchvision.transforms.CenterCrop(resize),
                torchvision.transforms.ToTensor()])

            categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
            train_data = VisADataset(root_path, transform=transform, categories=categories, phase='normal')
            test_data = VisADataset(root_path, transform=transform, categories=categories, phase='anomaly')
        
        else:
            raise NotImplementedError("Not impelemented dataset!")
        
    # Create sub classes
    if args.one_class_idx != None:
        if args.dataset == 'cifar100':
            train_data.targets = sparse2coarse(train_data.targets)
            test_data.targets = sparse2coarse(test_data.targets)
            train_data_in = get_subclass_dataset(train_data, args.one_class_idx)
            test_data_in = get_subclass_dataset(test_data, args.one_class_idx)
        else:
            train_data_in = get_subclass_dataset(train_data, args.one_class_idx)
            test_data_in = get_subclass_dataset(test_data, args.one_class_idx)
    
    return train_data_in, test_data_in, train_data, test_data


def calculate_accuracy(model_output, true_labels):
    # Convert sigmoid output to class predictions
    predicted_classes = [1 if x > 0.5 else 0 for x in model_output]
    
    # Flatten the tensor if necessary
    if len(predicted_classes) != len(true_labels):
        predicted_classes = torch.tensor(predicted_classes).view(-1)
        
    accuracy = accuracy_score(true_labels, predicted_classes)
    return accuracy


def novelty_detection(eval_in, eval_out, train_features_in, mahal_thresh, net, args):

    net = net.to(args.device)
    net.eval()  # enter valid mode

    pred_in = []
    pred_out = []
    targets_in_list = []
    targets_out_list = []
    in_features_list = []
    out_features_list = []
    
    model_preds = []
    trues = []
    with torch.no_grad():
        for data_in, data_out in zip(eval_in, eval_out):       
            inputs_in, targets_in = data_in
            inputs_out, targets_out = data_out
            
            inputs_in , inputs_out = inputs_in.to(args.device) , inputs_out.to(args.device)

            preds_in, normal_features = net(inputs_in, True)
            preds_out, out_features = net(inputs_out, True)
            
            in_features_list.extend(normal_features[-1].detach().cpu().numpy())
            out_features_list.extend(out_features[-1].detach().cpu().numpy())

            model_preds.extend(torch.cat([preds_in, preds_out]).detach().cpu().numpy())
            trues.extend(torch.cat([torch.zeros_like(preds_in), torch.ones_like(preds_out)]).detach().cpu().numpy())
            
        accuracy = calculate_accuracy(model_preds, trues)

        in_features_list = torch.tensor(np.array(in_features_list))
        out_features_list = torch.tensor(np.array(out_features_list))
        l1 = torch.zeros(in_features_list.shape[0])
        l2 = torch.ones(out_features_list.shape[0])
        targets = torch.cat([l1, l2], dim=0)

                                                               
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
                preds = torch.cat([torch.tensor(pred_in), torch.tensor(pred_out)], dim=0)
                preds = (preds + 1)/2
                auc = roc_auc_score(to_np(targets), to_np(preds))
        elif args.score == 'knn':
            f_list = torch.cat([in_features_list, out_features_list], dim=0)
            distances = knn_score(to_np(train_features_in), to_np(f_list))
            auc = roc_auc_score(targets, distances)
    return auc, accuracy


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

    if args.img_size == 224:
        if args.model == 'resnet18':
            model = resnet18(num_classes=args.num_classes)
        elif args.model == 'resnet50':
            model = resnet50(num_classes=args.num_classes)
        else:
            raise NotImplementedError("Not implemented model!")
        
    elif args.img_size == 32:
        if args.model == 'resnet18':
            model = ResNet18(num_classes=args.num_classes)
        elif args.model == 'resnet50':
            model = ResNet50(num_classes=args.num_classes)
        else:
            raise NotImplementedError("Not implemented model!")
    else:
        raise NotImplementedError("Not implemented image size!")                
    
    if args.model_path:
        m = torch.load(args.model_path, weights_only=True)
        model.load_state_dict(m)

    return model


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


if __name__ == "__main__":

    args = parsing()
    os.environ['CUDA_VISIBLE_DEVICES']="0,1"
    torch.manual_seed(args.seed)

    
    if args.device == 'cuda':
        args.device=f'cuda:{args.gpu}'

    model = load_model(args)

    root_path = 'D:/Datasets/data/'
    args.last_lr = args.learning_rate
    in_distance = torch.tensor(0)

    train_data_in, test_data_in, train_data, test_data = load_data(root_path, args)

    train_data_in_loader = DataLoader(train_data_in, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_in_loader = DataLoader(test_data_in, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.score == 'svm':
        from sklearn.svm import OneClassSVM
        svm_model = OneClassSVM(kernel='sigmoid', gamma='auto', nu=0.2)
        train_features_in = feature_extraction(train_data_in_loader, model)
        print(f"Fitting svm model {train_features_in.shape}")
        svm_model.fit(train_features_in.detach().cpu().numpy())
        args.svm_model = svm_model
    
    elif args.score == 'knn':
        train_features_in = feature_extraction(train_data_in_loader, model)

    elif args.score == 'mahalanobis':
        train_features_in = feature_extraction(train_data_in_loader, model)
        mean, inv_covariance, features = mahalanobis_parameters(train_features_in, model, args)

    else:
        raise NotImplementedError("Not implemented score!")

    resutls = {}
    aucs = []
    accs = []

    # resutls['OD']=in_distance.detach().cpu().numpy()
    if args.dataset == 'cifar100':
        args.classes = 20
    elif args.dataset == 'mvtec_ad':
        args.classes = 15
    else:
        args.classes = 10
        
    if args.one_class_idx != None:
        for id in range(args.classes):
            if id == args.one_class_idx:
                continue
            
            test_data_out = get_subclass_dataset(test_data, id)
            test_data_out_loader = DataLoader(test_data_out, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

            auc, acc = novelty_detection(test_data_in_loader, test_data_out_loader, train_features_in, in_distance + args.thresh, model, args)
            aucs.append(auc)
            accs.append(acc)

            print(f"Evaluation distance on class {id}: auc: {auc}, acc: {acc}")
    else:
        test_data_out_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
        auc, acc = novelty_detection(test_data_in_loader, test_data_out_loader, train_features_in, in_distance + args.thresh, model, args)

        aucs.append(auc)
        accs.append(acc)
        
        print(f"Evaluation distance on datases auc: {auc}, acc: {acc}")


    print(f"Average auc is: {np.mean(aucs)}, acc: {np.mean(accs)}")
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
