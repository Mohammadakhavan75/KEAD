import os
import math
import argparse
import numpy as np
import torch
import faiss
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset_loader import (
    load_np_dataset, 
    MVTecADDataset, 
    VisADataset, 
    SVHN, 
    get_subclass_dataset, 
    sparse2coarse
)
from models.resnet import ResNet18, ResNet50

# Constant for CIFAR100 superclasses
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


def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array."""
    return x.cpu().numpy()


def parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description='Model evaluation parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    # Optimizer parameters
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
    # Dataset and evaluation parameters
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
    parser.add_argument('--num_classes', default=1, type=int, help='noise')
    
    args = parser.parse_args()

    return args

def normalize_features(features, epsilon=1e-12):
    """
    Normalize the features along the rows (L2 normalization).
    
    Args:
        features (np.ndarray): Input feature array of shape (num_samples, feature_dim).
        epsilon (float): Small constant to avoid division by zero.
        
    Returns:
        np.ndarray: Normalized features.
    """
    norms = torch.linalg.norm(features, axis=1, keepdims=True)
    return features / (norms + epsilon)

def knn_score(train_set: np.ndarray, test_set: np.ndarray, n_neighbors: int = 1) -> np.ndarray:
    """
    Compute KNN score using Faiss.
    
    Returns the sum of squared L2 distances to the n_neighbors nearest neighbors.
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    # index = faiss.IndexFlatIP(train_set.shape[1])
    index.add(train_set)
    distances, _ = index.search(test_set, n_neighbors)
    return np.sum(distances, axis=1)


def compute_mahalanobis_parameters(features: torch.Tensor, device: str):
    """
    Compute the mean vector and inverse covariance matrix for the features.
    
    Args:
        features: Tensor of shape (N, D).
        device: Device string.
    Returns:
        mean, inv_covariance (both torch.Tensors) and the original features.
    """
    print("Calculating Mahalanobis parameters...")
    mean = features.mean(dim=0)
    cov = torch.cov(features.t())
    # Add a small constant for numerical stability
    cov += 1e-12 * torch.eye(cov.size(0), device=device)
    inv_covariance = torch.linalg.inv(cov)
    return mean, inv_covariance, features

def mahalanobis_distance(x: torch.Tensor, mu: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mahalanobis distance of vector x from mean mu.
    """
    centered = x - mu
    # Compute distance: sqrt( x^T inv_cov x )
    dist = torch.sqrt(torch.clamp(torch.matmul(centered.unsqueeze(0), torch.matmul(inv_cov, centered.unsqueeze(1))), min=0))
    return dist


def load_data(root_path: str, args):
    """
    Load train and test datasets based on the provided arguments.
    
    If noise is provided, load noisy test data; otherwise, load the original data.
    If a one-class index is provided, extract the corresponding subclass.
    
    Returns:
        (train_in, test_in, train_full, test_full)
    """

    test_transform = Compose([ToTensor()])
    if args.noise:
        print(f"Loading noise {args.noise} as test data")
        if args.dataset == 'cifar10':
            np_test_img_path = os.path.join(root_path, 'generalization_repo_dataset', 'cifar10_Test_s5', f'{args.noise}.npy')
            np_test_target_path = os.path.join(root_path, 'generalization_repo_dataset', 'cifar10_Test_s5', 'labels.npy')
            train_data = torchvision.datasets.CIFAR10(root_path, train=True, transform=test_transform, download=True)
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10')
        elif args.dataset == 'svhn':
            np_test_img_path = os.path.join(root_path, 'generalization_repo_dataset', 'svhn_Test_s5', f'{args.noise}.npy')
            np_test_target_path = os.path.join(root_path, 'generalization_repo_dataset', 'svhn_Test_s5', 'labels.npy')
            train_data = SVHN(root=root_path, split="train", transform=test_transform)
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='svhn')
        elif args.dataset == 'cifar100':
            np_test_img_path = os.path.join(root_path, 'generalization_repo_dataset', 'cifar100_Test_s5', f'{args.noise}.npy')
            np_test_target_path = os.path.join(root_path, 'generalization_repo_dataset', 'cifar100_Test_s5', 'labels.npy')
            train_data = torchvision.datasets.CIFAR100(root_path, train=True, transform=test_transform, download=True)
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar100')
        elif args.dataset == 'mvtec_ad':
            resize=args.img_size
            transform = Compose([
                Resize(math.ceil(resize * 1.14)),
                CenterCrop(resize),
                ToTensor()])    
            categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
            train_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='train')
            np_test_img_path = os.path.join(root_path, 'generalization_repo_dataset', 'mvtec_ad_Test_s5', f'{args.noise}.npy')
            np_test_target_path = os.path.join(root_path, 'generalization_repo_dataset', 'mvtec_ad_Test_s5', 'labels.npy')
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='mvtec_ad')
        elif args.dataset == 'visa':
            import math
            resize=args.img_size
            transform = Compose([
                Resize(math.ceil(resize*1.14)),
                CenterCrop(resize),
                ToTensor()])
            categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
            train_data = VisADataset(root_path, transform=transform, categories=categories, phase='normal')
            np_test_img_path = os.path.join(root_path, 'generalization_repo_dataset', 'visa_Train_s5', f'{args.noise}.npy')
            np_test_target_path = os.path.join(root_path, 'generalization_repo_dataset', 'visa_Train_s5', 'labels.npy')
            test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='visa')
        else:
            raise NotImplementedError("Not impelemented dataset!")
    else:
        print("Loading original test data")
        if args.dataset == 'cifar10':
            train_data = torchvision.datasets.CIFAR10(root_path, train=True, transform=test_transform, download=True)
            test_data = torchvision.datasets.CIFAR10(root_path, train=False, transform=test_transform, download=True)
        elif args.dataset == 'svhn':
            train_data = SVHN(root=root_path, split="train", transform=test_transform)
            test_data = SVHN(root=root_path, split="test", transform=test_transform)
        elif args.dataset == 'cifar100':
            train_data = torchvision.datasets.CIFAR100(
                root_path, train=True, transform=test_transform, download=True)
            test_data = torchvision.datasets.CIFAR100(
                root_path, train=False, transform=test_transform, download=True)
        elif args.dataset == 'mvtec_ad':
            resize=args.img_size
            transform = Compose([
                Resize(math.ceil(resize * 1.14)),
                CenterCrop(resize),
                ToTensor()])  
            categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
            train_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='train')
            test_data = MVTecADDataset(root_path, transform=transform, categories=categories, phase='test')
        elif args.dataset == 'visa':
            resize=args.img_size
            transform = Compose([
                Resize(math.ceil(resize * 1.14)),
                CenterCrop(resize),
                ToTensor()])
            categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
            train_data = VisADataset(root_path, transform=transform, categories=categories, phase='normal')
            test_data = VisADataset(root_path, transform=transform, categories=categories, phase='anomaly')
        else:
            raise NotImplementedError("Not impelemented dataset!")
        
    # If a one-class index is specified, extract the subclass; otherwise use full datasets.
    if args.one_class_idx is not None:
        if args.dataset == 'cifar100':
            train_data.targets = sparse2coarse(train_data.targets)
            test_data.targets = sparse2coarse(test_data.targets)

        train_in = get_subclass_dataset(train_data, args.one_class_idx)
        test_in = get_subclass_dataset(test_data, args.one_class_idx)
    else:
        train_in, test_in = train_data, test_data
    return train_in, test_in, train_data, test_data


def calculate_accuracy(model_output, true_labels):
    """
    Calculate classification accuracy given model outputs.
    
    Assumes sigmoid outputs and a threshold of 0.5.
    """
    predicted = [1 if x > 0.5 else 0 for x in model_output]
    if len(predicted) != len(true_labels):
        predicted = torch.tensor(predicted).view(-1)
    return accuracy_score(true_labels, predicted)


def novelty_detection(eval_in_loader, eval_out_loader, train_features_in, threshold, net, args,
                      mean: torch.Tensor = None, inv_covariance: torch.Tensor = None):
    """
    Evaluate novelty detection performance on in- and out-of-distribution data.
    
    Depending on args.score, compute anomaly scores using Mahalanobis, SVM, or KNN.
    Returns the AUC and classification accuracy.
    """
    net = net.to(args.device)
    net.eval()

    in_features_list = []
    out_features_list = []
    model_preds = []
    true_labels = []

    targets_in_list = []
    targets_out_list = []


    with torch.no_grad():
        for (inputs_in, targets_in), (inputs_out, targets_out) in zip(eval_in_loader, eval_out_loader):
            inputs_in = inputs_in.to(args.device)
            inputs_out = inputs_out.to(args.device)

            inputs_in , inputs_out = inputs_in.to(args.device) , inputs_out.to(args.device)

            preds_in, features_in = net(inputs_in)
            preds_out, features_out = net(inputs_out)

            # Save features from the last layer (assumed to be at index -1)
            in_features_list.extend(features_in[-1].detach().cpu().numpy())
            out_features_list.extend(features_out[-1].detach().cpu().numpy())

            # Collect model predictions (for accuracy) and true labels
            model_preds.extend(torch.cat([preds_in, preds_out]).detach().cpu().numpy())
            true_labels.extend(torch.cat([torch.zeros_like(preds_in), torch.ones_like(preds_out)]).detach().cpu().numpy())
            targets_in_list.extend(targets_in.cpu().numpy())
            targets_out_list.extend(targets_out.cpu().numpy())

    # Calculate classification accuracy from network outputs
    # accuracy_val = calculate_accuracy(model_preds, true_labels)
    accuracy_val = 0.0
        
        # in_features_list = torch.tensor(np.array(in_features_list))
        # out_features_list = torch.tensor(np.array(out_features_list))
        # l1 = torch.zeros(in_features_list.shape[0])
        # l2 = torch.ones(out_features_list.shape[0])
        # targets = torch.cat([l1, l2], dim=0)
    
    # Depending on the scoring method, compute anomaly AUC
    if args.score == 'mahalanobis':
        if mean is None or inv_covariance is None:
            raise ValueError("Mean and inverse covariance must be provided for Mahalanobis scoring.")
        pred_in = [mahalanobis_distance(torch.tensor(feat), mean, inv_covariance).item() < threshold
                   for feat in in_features_list]
        pred_out = [mahalanobis_distance(torch.tensor(feat), mean, inv_covariance).item() < threshold
                    for feat in out_features_list]
        preds = np.concatenate([pred_in, pred_out])
        targets = np.concatenate([np.zeros(len(pred_in)), np.ones(len(pred_out))])
        auc = roc_auc_score(targets, preds)

    elif args.score == 'svm':
        # SVM scoring branch
        if args.dataset == 'anomaly':
            f_list = torch.cat([
                torch.tensor(in_features_list),
                torch.tensor(out_features_list)
            ], dim=0)
            preds = args.svm_model.predict(f_list.detach().cpu().numpy())
            preds = (preds + 1) / 2
            targets = np.concatenate([np.array(targets_in_list), np.array(targets_out_list)])
            auc = roc_auc_score(targets, preds)
        else:
            # in_features_list = normalize_features(torch.tensor(np.array(in_features_list)))
            # out_features_list = normalize_features(torch.tensor(np.array(out_features_list)))
            pred_in = (args.svm_model.predict(np.array(in_features_list)) == 1).astype(int)
            pred_out = (args.svm_model.predict(np.array(out_features_list)) == -1).astype(int)
            preds = np.concatenate([pred_in, pred_out])
            preds = (preds + 1) / 2
            targets = np.concatenate([np.zeros(len(pred_in)), np.ones(len(pred_out))])
            auc = roc_auc_score(targets, preds)

    elif args.score == 'knn':
        f_list = torch.cat([torch.tensor(np.array(in_features_list)), torch.tensor(np.array(out_features_list))], axis=0)

        # train_features_in = normalize_features(train_features_in)
        # f_list = normalize_features(f_list)

        distances = knn_score(to_np(train_features_in), to_np(f_list))
        targets = np.concatenate([np.zeros(len(in_features_list)), np.ones(len(out_features_list))])
        auc = roc_auc_score(targets, distances)

    else:
        raise NotImplementedError("Scoring method not implemented!")

    return auc, accuracy_val



def eval_auc_anomaly(loader, train_features_in, net, args):
    """
    Evaluate anomaly detection AUC on a given loader.
    """
    net = net.to(args.device)
    net.eval()

    features_list = []
    anomaly_labels = []

    with torch.no_grad():
        for inputs, _, anomaly in loader:
            anomaly_labels.extend(anomaly)
            inputs = inputs.to(args.device)
            _, features = net(inputs, return_features=True)
            features_list.append(features[-1])
    features_list = torch.cat(features_list, dim=0)

    if args.score == 'svm':
        preds = args.svm_model.predict(features_list.detach().cpu().numpy())
        preds = (preds + 1) / 2
        # Invert anomaly labels if needed
        anomaly_labels = [1 - x for x in anomaly_labels]
        auc = roc_auc_score(anomaly_labels, preds)
    elif args.score == 'knn':
        distances = knn_score(to_np(train_features_in), to_np(features_list))
        auc = roc_auc_score(anomaly_labels, distances)
    else:
        raise NotImplementedError("Scoring method not implemented in eval_auc_anomaly!")

    return auc


def load_model(args):
    """
    Load the model based on image size and model type.
    """
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


def feature_extraction(loader, net, device: str):
    """
    Extract features from the last layer for all inputs in the loader.
    """
    print("Extracting features...")
    net = net.to(device)
    net.eval()
    features = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _, feats = net(inputs)
            features.append(feats[-1])
    
    return torch.cat(features, dim=0)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']="0,1"
    torch.manual_seed(args.seed)

    # Set device
    if args.device == 'cuda':
        args.device=f'cuda:{args.gpu}'

    model = load_model(args)
    root_path = 'D:/Datasets/data/'
    args.last_lr = args.learning_rate
    in_distance = torch.tensor(0)

    # Load datasets; if no one_class_idx is provided, in-distribution equals full dataset.
    train_in, test_in, train_full, test_full = load_data(root_path, args)


    train_in_loader = DataLoader(train_in, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_in_loader = DataLoader(test_in, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Extract features and prepare scoring method
    if args.score == 'svm':
        from sklearn.svm import OneClassSVM
        train_features_in = feature_extraction(train_in_loader, model, args.device)
        print(f"Fitting SVM model with features shape: {train_features_in.shape}")
        svm_model = OneClassSVM(kernel='sigmoid', gamma='auto', nu=0.2)
        # train_features_in = normalize_features(train_features_in)
        svm_model.fit(train_features_in.cpu().numpy())
        args.svm_model = svm_model
    elif args.score == 'knn':
        train_features_in = feature_extraction(train_in_loader, model, args.device)
    elif args.score == 'mahalanobis':
        train_features_in = feature_extraction(train_in_loader, model, args.device)
        mean, inv_covariance, _ = compute_mahalanobis_parameters(train_features_in, args.device)
    else:
        raise NotImplementedError("Scoring method not implemented!")

    aucs = []
    accs = []
    
    # Set number of classes based on dataset
    if args.dataset == 'cifar100':
        args.classes = 20
    elif args.dataset == 'mvtec_ad':
        args.classes = 15
    else:
        args.classes = 10
        
    # Evaluate novelty detection
    if args.one_class_idx is not None:
        # Evaluate on each class that is not the in-distribution class
        for class_id in range(args.classes):
            if class_id == args.one_class_idx:
                continue
            test_out = get_subclass_dataset(test_full, class_id)
            test_out_loader = DataLoader(test_out, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
            auc, acc = novelty_detection(test_in_loader, test_out_loader, train_features_in, in_distance.item() + args.thresh, model, args, mean=mean if args.score == 'mahalanobis' else None,
                                         inv_covariance=inv_covariance if args.score == 'mahalanobis' else None)
            aucs.append(auc)
            accs.append(acc)
            print(f"Class {class_id} -> AUC: {auc}, Accuracy: {acc}")
    else:
        test_out_loader = DataLoader(test_full, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
        auc, acc = novelty_detection(test_in_loader, test_out_loader, train_features_in,
                                     in_distance.item() + args.thresh, model, args,
                                     mean=mean if args.score == 'mahalanobis' else None,
                                     inv_covariance=inv_covariance if args.score == 'mahalanobis' else None)
        aucs.append(auc)
        accs.append(acc)
        print(f"Evaluation on dataset -> AUC: {auc}, Accuracy: {acc}")


    avg_auc = np.mean(aucs)
    avg_acc = np.mean(accs)
    print(f"Average AUC: {avg_auc}, Average Accuracy: {avg_acc}")
    print(f"Results from model: {args.model_path}")
    print(f"In-class index: {args.one_class_idx}")
    results = {'avg_auc': avg_auc}

    if args.csv:
        import pandas as pd
        df = pd.DataFrame(results, index=[0])
        # Construct save path based on model_path components
        model_parts = args.model_path.split('_')
        ev = model_parts[1] if len(model_parts) > 1 else "default"
        save_dir = os.path.join('.', 'csv_results', ev, f'dataset_{args.dataset}', f'report_class_{args.one_class_idx}')
        os.makedirs(save_dir, exist_ok=True)
        csv_filename = os.path.join(save_dir, os.path.basename(os.path.dirname(args.model_path)) + ".csv")
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()