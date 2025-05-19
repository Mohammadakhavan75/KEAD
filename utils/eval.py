import faiss
import numpy as np
import torch
from dataset_loader import get_dataset, get_subclass_dataset
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
import torchvision
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F

def to_np(x):
    return x.data.cpu().numpy()



def knn_score_calculation(train_set, test_set, n_neighbours=1):
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def novelty_detection(eval_in, eval_out, train_features_in, net, args):
    net = net.to(args.device)
    net.eval()  # enter valid mode

    in_features_list = []
    out_features_list = []
    
    model_preds = []
    trues = []
    with torch.no_grad():
        for (inputs_in, _), (inputs_out, _) in zip(eval_in, eval_out):
            inputs_in = inputs_in.to(args.device)
            inputs_out = inputs_out.to(args.device)

            _, normal_features = net(inputs_in)
            _, out_features = net(inputs_out)
            
            in_features_list.extend(normal_features[-1].detach().cpu().numpy())
            out_features_list.extend(out_features[-1].detach().cpu().numpy())

            
        in_features_list = F.normalize(torch.tensor(np.array(in_features_list)), dim=1)
        out_features_list = F.normalize(torch.tensor(np.array(out_features_list)), dim=1)
        # Prepare labels
        targets = torch.cat([
            torch.zeros(in_features_list.shape[0]),
            torch.ones(out_features_list.shape[0])
        ], dim=0)

        f_list = torch.cat([in_features_list, out_features_list], dim=0)
        train_features_in = F.normalize(train_features_in, dim=1)
        distances = knn_score_calculation(to_np(train_features_in), to_np(f_list))
        auc = roc_auc_score(targets, distances)
    return auc


def feature_extraction(loader, net, args):
    print("extracting features...")
    net = net.to(args.device)
    net.eval()  # enter eval mode
    
    try:
        features = []
        with torch.no_grad():
            for data_in in loader:
                inputs_in, _ = data_in
                inputs = inputs_in.to(args.device)
                
                _, normal_features = net(inputs)            
                # Move features to CPU immediately to free GPU memory
                features.append(normal_features[-1].cpu())
                
                # Clear any unused memory
                del inputs
                del normal_features
                torch.cuda.empty_cache()
        
        features = torch.cat(features, dim=0)
        return features
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()


def eval_cifar10_novelity(model, args, root_path):
    try:
        test_transform = v2.Compose([v2.ToTensor()])
        train_data, test_data = get_dataset(args, root_path, test_transform)
        
        train_data_in = get_subclass_dataset(train_data, args.one_class_idx)
        test_data_in = get_subclass_dataset(test_data, args.one_class_idx)
        
        train_in_loader = DataLoader(train_data_in, shuffle=True, batch_size=args.batch_size, pin_memory=True)
        test_in_loader = DataLoader(test_data_in, shuffle=False, batch_size=args.batch_size, pin_memory=True)
        
        train_features_in = feature_extraction(train_in_loader, model, args)

        aucs = []
        for id in range(10):
            if id == args.one_class_idx:
                continue

            test_data_out = get_subclass_dataset(test_data, id)
            test_out_loader = DataLoader(test_data_out, shuffle=False, batch_size=args.batch_size, pin_memory=True)
            
            try:
                auc = novelty_detection(test_in_loader, test_out_loader, train_features_in, model, args)
                aucs.append(auc)
                print(f"Evaluation distance on class {id}: auc: {auc}")
            except Exception as e:
                print(f"Error processing class {id}: {str(e)}")
                continue
            finally:
                # Cleanup after each iteration
                del test_data_out
                del test_out_loader

        return np.mean(aucs)
    except Exception as e:
        print(f"Error in eval_cifar10_novelity: {str(e)}")
        raise
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        if 'train_features_in' in locals():
            del train_features_in


def test_novelty(backbone, args):
    device = args.device
    backbone.eval()

    try:
        # prepare train embeddings
        train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
        normal_class = args.normal_class
        train_norm = [(img, label) for img, label in train_full if label == normal_class]
        ganchor = torch.stack([v2.ToTensor()(img) for img, _ in train_norm])
        ganchor = v2.Normalize((0.5,), (0.5,))(ganchor).to(device)
        
        with torch.no_grad():
            Z_train = F.normalize(backbone(ganchor), dim=1).cpu().numpy()

        # fit one-class SVM
        oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
        oc_svm.fit(Z_train)

        # prepare test set: all classes
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=v2.Compose([
            v2.Resize(32), v2.ToTensor(), v2.Normalize((0.5,), (0.5,))
        ]))
        
        # Use a context manager for the DataLoader
        with DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True) as test_loader:
            # compute scores
            y_true, y_scores = [], []
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.to(device)
                with torch.no_grad():
                    z = F.normalize(backbone(x), dim=1).cpu().numpy()
                scores = -oc_svm.decision_function(z)  # higher == more anomalous
                y_true.extend([0 if lbl == normal_class else 1 for lbl in y.numpy()])
                y_scores.extend(scores)

        auc = roc_auc_score(y_true, y_scores)
        print(f"Test AUROC: {auc:.4f}")
        
    except Exception as e:
        print(f"Error during novelty testing: {str(e)}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        if hasattr(test_loader, 'dataset'):
            del test_loader
        if 'Z_train' in locals():
            del Z_train
        if 'ganchor' in locals():
            del ganchor
