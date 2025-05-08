from utils.parser import args_parser
from utils.paths import create_path
from utils.loader import load_model
import numpy as np
from tqdm import tqdm
import random
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.utils import augmentation_layers as augl

from sklearn.metrics import roc_auc_score


##########################
import faiss
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset_loader import load_np_dataset, get_subclass_dataset
from torchvision.transforms import transforms
import torchvision



from dataset_loader import noise_loader, load_cifar10,\
     load_svhn, load_cifar100, load_imagenet, load_mvtec_ad, load_visa

# -------------------------------
# Dataset with per-image K transforms
# -------------------------------
class KEPerImageDataset(Dataset):
    def __init__(self, dataset, K, transform_pool, base_preprocess):
        self.dataset = dataset          # torchvision dataset
        self.K = K                      # number of stochastic views
        self.transform_pool = transform_pool
        self.base_preprocess = base_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # clean anchor
        anchor = self.base_preprocess(x)
        views = []
        ids = []
        for _ in range(self.K):
            t_idx = np.random.randint(len(self.transform_pool))
            views.append(self.transform_pool[t_idx](x))
            ids.append(t_idx)
        views = torch.stack(views)
        ids = torch.tensor(ids, dtype=torch.long)
        return anchor, views, ids, y

# -------------------------------
# Running stats strategy for pos/neg masks
# -------------------------------
class RunningStatsStrategy:
    def __init__(self, n_transforms, alpha=0.7, momentum=0.01):
        self.mu = torch.zeros(n_transforms)
        self.var = torch.ones(n_transforms)
        self.alpha = alpha
        self.momentum = momentum

    @torch.no_grad()
    def update_and_get_mask(self, sim, ids):
        # sim: (B, K), ids: (B, K)
        B, K = sim.shape
        mu_t = self.mu[ids]          # (B,K)
        sigma_t = torch.sqrt(self.var[ids] + 1e-6)
        # positive mask
        pos_mask = sim > (mu_t - self.alpha * sigma_t)
        # update stats
        sim_flat = sim.flatten()
        ids_flat = ids.flatten()
        for t in ids_flat.unique():
            mask = ids_flat == t
            s_batch = sim_flat[mask]
            if s_batch.numel() > 0:
                m_batch = s_batch.mean()
                v_batch = s_batch.var(unbiased=False)
                self.mu[t] = (1 - self.momentum) * self.mu[t] + self.momentum * m_batch
                self.var[t] = (1 - self.momentum) * self.var[t] + self.momentum * v_batch
        return pos_mask

# -------------------------------
# Multi-positive InfoNCE loss
# -------------------------------
def info_nce_multi(z_anchor, z_views, pos_mask, tau=0.2):
    # z_anchor: (B, D), z_views: (B, K, D), pos_mask: (B, K)
    B, K, D = z_views.shape
    loss = 0.0
    valid = 0
    for i in range(B):
        sim = F.cosine_similarity(z_anchor[i].unsqueeze(0), z_views[i], dim=1) / tau
        pos_inds = pos_mask[i].nonzero(as_tuple=False).flatten()
        if pos_inds.numel() == 0:
            continue
        valid += 1
        pos_sims = sim[pos_inds]
        exp_all = torch.exp(sim)
        denom = exp_all.sum()
        loss_i = - (1.0 / pos_inds.numel()) * torch.log(torch.exp(pos_sims).sum() / denom)
        loss += loss_i
    if valid > 0:
        loss = loss / valid
    return loss

# -------------------------------
# Feature extractor wrapper
# -------------------------------
def get_backbone_embedding(model, x):
    # returns l2-normalized embedding
    with torch.no_grad():
        z = model(x)
        z = F.normalize(z, dim=1)
    return z

# -------------------------------
# Main training & test routines
# -------------------------------
def train_contrastive(stats, model, train_loader, optimizer, transform_pool, epoch, scheduler, args, train_global_iter, writer):
    device = args.device
    N = len(transform_pool)
    # training
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for anchor, _ in pbar:
        anchor = anchor.to(device)
        B = anchor.size(0)
        ids = torch.randint(0, N, size=(args.k_view,))
        
        views = torch.vstack([transform_pool[ids].to(device)(anchor)])
        views = views.to(device).view(-1, 3, 32, 32)

        optimizer.zero_grad()
        # get embeddings
        z_a = F.normalize(model(anchor), dim=1)            # (B,D)
        z_v = F.normalize(model(views), dim=1).view(B, args.K, -1)  # (B,K,D)

        # cosine similarities
        sim = torch.einsum('bd,bkd->bk', z_a, z_v)

        # get pos mask
        pos_mask = stats.update_and_get_mask(sim.cpu(), ids.cpu()).to(device)

        # compute loss
        loss = info_nce_multi(z_a, z_v, pos_mask, tau=args.tau)

        
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())


        # writer.add_scalar("Train/loss", loss.item(), train_global_iter)
        # writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
        # writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)
        # writer.add_scalar("Train/norm_p", torch.mean(data_norm).detach().cpu().numpy(), train_global_iter)
        # writer.add_scalar("Train/norm_n", torch.mean(negative_norms).detach().cpu().numpy(), train_global_iter)
        # writer.add_scalar("Train/norm_d", torch.mean(positive_norms).detach().cpu().numpy(), train_global_iter)
        train_global_iter += 1


    return backbone


def test_novelty(backbone, args):
    device = args.device
    backbone.eval()

    # prepare train embeddings
    train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    normal_class = args.normal_class
    train_norm = [ (img, label) for img, label in train_full if label == normal_class]
    ganchor = torch.stack([ transforms.ToTensor()(img) for img, _ in train_norm ])
    ganchor = transforms.Normalize((0.5,), (0.5,))(ganchor).to(device)
    with torch.no_grad():
        Z_train = F.normalize(backbone(ganchor), dim=1).cpu().numpy()

    # fit one-class SVM
    oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    oc_svm.fit(Z_train)

    # prepare test set: all classes
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ]))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

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


def loading_datasets(args, data_path, imagenet_path):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        train_loader, test_loader = load_cifar10(data_path, 
                                            batch_size=args.batch_size,
                                            one_class_idx=args.one_class_idx,
                                            seed=args.seed)
    elif args.dataset == 'svhn':
        args.num_classes = 10
        train_loader, test_loader = load_svhn(data_path, 
                                            batch_size=args.batch_size,
                                            one_class_idx=args.one_class_idx,
                                            seed=args.seed)
    elif args.dataset == 'cifar100':
        args.num_classes = 20
        train_loader, test_loader = load_cifar100(data_path, 
                                                batch_size=args.batch_size,
                                                one_class_idx=args.one_class_idx,
                                                seed=args.seed)
    elif args.dataset == 'imagenet30':
        args.num_classes = 30
        train_loader, test_loader = load_imagenet(imagenet_path, 
                                                batch_size=args.batch_size,
                                                one_class_idx=args.one_class_idx,
                                                seed=args.seed)
    elif args.dataset == 'mvtec_ad':
        args.num_classes = 15
        train_loader, test_loader = load_mvtec_ad(data_path, 
                                                resize=args.img_size,
                                                batch_size=args.batch_size,
                                                one_class_idx=args.one_class_idx,
                                                seed=args.seed)

    elif args.dataset == 'visa':
        args.num_classes = 12
        train_loader, test_loader = load_visa(data_path, 
                                                resize=args.img_size,
                                                batch_size=args.batch_size,
                                                one_class_idx=args.one_class_idx,
                                                seed=args.seed)
    
    return train_loader, test_loader


def set_seed(seed_nu):
    torch.manual_seed(seed_nu)
    random.seed(seed_nu)
    np.random.seed(seed_nu)


def novelty_detection(eval_in, eval_out, train_features_in, net, args):
    net = net.to(args.device)
    net.eval()  # enter valid mode

    in_features_list = []
    out_features_list = []
    
    model_preds = []
    trues = []
    with torch.no_grad():
        for data_in, data_out in zip(eval_in, eval_out):
            inputs_in, targets_in = data_in
            inputs_out, targets_out = data_out
            
            inputs_in , inputs_out = inputs_in.to(args.device) , inputs_out.to(args.device)

            preds_in, normal_features = net(inputs_in)
            preds_out, out_features = net(inputs_out)
            
            in_features_list.extend(normal_features[-1].detach().cpu().numpy())
            out_features_list.extend(out_features[-1].detach().cpu().numpy())

            
        in_features_list = torch.tensor(np.array(in_features_list))
        out_features_list = torch.tensor(np.array(out_features_list))
        l1 = torch.zeros(in_features_list.shape[0])
        l2 = torch.ones(out_features_list.shape[0])
        targets = torch.cat([l1, l2], dim=0)
    
        f_list = torch.cat([in_features_list, out_features_list], dim=0)
        distances = knn_score(to_np(train_features_in), to_np(f_list))
        auc = roc_auc_score(targets, distances)
    return auc


def eval_cifar10_novelity(model, args, root_path):
    np_test_img_path = root_path + f'/generalization_repo_dataset/cifar10_Test_s5/rot270.npy'
    np_test_target_path = root_path + '/generalization_repo_dataset/cifar10_Test_s5/labels.npy'
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(root_path, train=True, transform=test_transform, download=True)
    test_data = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset='cifar10')
    train_data_in = get_subclass_dataset(train_data, args.one_class_idx)
    test_data_in = get_subclass_dataset(test_data, args.one_class_idx)
    train_data_in_loader = DataLoader(train_data_in, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_in_loader = DataLoader(test_data_in, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    train_features_in = feature_extraction(train_data_in_loader, model)

    aucs = []
    for id in range(10):
        if id == args.one_class_idx:
            continue

        test_data_out = get_subclass_dataset(test_data, id)
        test_data_out_loader = DataLoader(test_data_out, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
        auc = novelty_detection(test_data_in_loader, test_data_out_loader, train_features_in, model, args)
        aucs.append(auc)
        print(f"Evaluation distance on class {id}: auc: {auc}")

    print(f"Average auc is: {np.mean(aucs)}")
    return np.mean(aucs)


def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    args = args_parser()

    root_path = config['root_path']
    data_path = config['data_path']
    imagenet_path = config['imagenet_path']
    args.config = config
    best_loss = torch.inf
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    args.device=f'cuda:{args.gpu}'
    train_global_iter = 0
    args.last_lr = args.learning_rate


    model_save_path, save_path = create_path(args)
    writer = SummaryWriter(save_path)
    args.save_path = save_path

    train_loader, test_loader = loading_datasets(args, data_path, imagenet_path)
    model, optimizer, scheduler = load_model(args)

    model = model.to(args.device)

    transform_pool = augl.get_augmentation_list()

    stats = RunningStatsStrategy(n_transforms=len(transform_pool), alpha=args.alpha)

    train_global_iter = 0
    for epoch in range(0, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns, colapse_metrics =\
            train_contrastive(stats, model, train_loader, optimizer, transform_pool, epoch, scheduler, args, train_global_iter, writer)
        
        # writer.add_scalar("AVG_Train/sim_p", avg_sim_ps, epoch)
        # writer.add_scalar("AVG_Train/sim_n", avg_sim_ns, epoch)
        # writer.add_scalar("colapse_metrics/effective_rank", colapse_metrics['effective_rank'], epoch)
        # writer.add_scalar("colapse_metrics/norm_mean", colapse_metrics['norm_mean'], epoch)
        # writer.add_scalar("colapse_metrics/norm_var", colapse_metrics['norm_var'], epoch)
        # writer.add_scalar("colapse_metrics/pos_alignment", colapse_metrics['pos_alignment'], epoch)
        # writer.add_scalar("colapse_metrics/neg_alignment", colapse_metrics['neg_alignment'], epoch)
        # writer.add_scalar("colapse_metrics/uniformity", colapse_metrics['uniformity'], epoch)

        # writer.add_scalar("AVG_Train/avg_loss", np.mean(epoch_loss['loss']), epoch)
        # writer.add_scalar("AVG_Train/avg_loss_contrastive", np.mean(epoch_loss['contrastive']), epoch)
        # writer.add_scalar("Train/lr", args.last_lr, epoch)

        # accuracy = evaluate(test_loader, test_positives_loader, test_negetives_loader, model, args)
        # print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])}, Eval/avg_acc: {accuracy}")
        print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])}")
        if epoch % 10 == 9:
            avg_auc = eval_cifar10_novelity(model, args, root_path)
            # writer.add_scalar("Eval/avg_auc", avg_auc, epoch)

        if (epoch) % (args.epochs / 10) == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))


        if np.mean(epoch_loss['loss']) < best_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'best_params.pt'))
            best_loss = np.mean(epoch_loss['loss'])

        # last_sim_ps = avg_sim_ps
        # last_sim_ns = avg_sim_ns
        # args.seed += epoch
        # set_seed(args.seed)

    torch.save(model.state_dict(), os.path.join(save_path, 'last_params.pt'))

if __name__ == '__main__':
    main()
