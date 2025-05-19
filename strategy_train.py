from utils.parser import args_parser
from utils.paths import create_path
from utils.loader import load_model
from utils.eval import evaluation
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
import torchvision.transforms.v2 as v2
import pickle


from dataset_loader import get_loader


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
    def update_and_get_mask(self, sim):
        # sim: (B, K)
        B, K = sim.shape
        mu_t = self.mu       # (B,K)
        sigma_t = torch.sqrt(self.var + 1e-6)
        # positive mask
        pos_mask = sim > (mu_t - self.alpha * sigma_t) # (B,K)
        # update stats
        for t in range(K):
            s_batch = sim[:,t]
            if s_batch.numel() > 0:
                m_batch = s_batch.mean()
                v_batch = s_batch.var(unbiased=False)
                self.mu[t] = (1 - self.momentum) * self.mu[t] + self.momentum * m_batch
                self.var[t] = (1 - self.momentum) * self.var[t] + self.momentum * v_batch
        return pos_mask

# -------------------------------
# Multi-positive InfoNCE loss
# -------------------------------
def info_nce_multi(z_anchor, z_views, tau=0.2):
    # z_anchor: (B, D), z_views: (B, K, D)
    B, K, D = z_views.shape

    # Compute cosine similarities: (B, K)
    sim = torch.einsum('bd,bkd->bk', z_anchor, z_views) / tau
     
    # InfoNCE loss: -log (exp(sim_pos) / (exp(sim_pos) + exp(sim_neg)))
    sim_pos = sim[:, 0]  # (B,)
    sim_neg = sim[:, 1]  # (B,)

    logits = torch.stack([sim_pos, sim_neg], dim=1)  # (B, 2)
    labels = torch.zeros(z_anchor.size(0), dtype=torch.long, device=z_anchor.device)  # positives are at index 0

    loss = F.cross_entropy(logits, labels)
    return loss


# -------------------------------
# Main training & test routines
# -------------------------------
def train_contrastive(stats, model, train_loader, optimizer, transform_sequence, epoch, scheduler, args, train_global_iter, writer):
    device = args.device

    # training
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for anchor, _ in pbar:
        anchor = anchor.to(device)
        B = anchor.size(0)
        
        views = transform_sequence(anchor)
        optimizer.zero_grad()
        
        # get embeddings and normilize the output
        z_a = F.normalize(model(anchor)[0], dim=1)            # (B,D)
        z_v = F.normalize(model(views)[0].view(B, int(views.shape[0]/B), -1), dim=2)  # (B,K,D)

        # compute loss
        loss = info_nce_multi(z_a, z_v, tau=args.temperature)

        
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


    # return backbone



def set_seed(seed_nu):
    torch.manual_seed(seed_nu)
    random.seed(seed_nu)
    np.random.seed(seed_nu)



def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    args = args_parser()

    root_path = config['root_path']
    data_path = config['data_path']
    imagenet_path = config['imagenet_path']
    args.config = config
    best_loss = torch.inf
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    # args.device=f'cuda:{args.gpu}'
    train_global_iter = 0
    args.last_lr = args.learning_rate


    model_save_path, save_path = create_path(args)
    writer = SummaryWriter(save_path)
    args.save_path = save_path

    transform = v2.Compose([
        v2.RandomChoice([
            v2.RandomRotation(degrees=(-5, 5)),
            v2.RandomGrayscale(p=0.8),
            v2.RandomHorizontalFlip(p=0.8),
        ], p=[0.3, 0.3, 0.3]),
        v2.ToTensor()
    ])
    train_loader, test_loader = get_loader(args, data_path, imagenet_path, transform)
    model, optimizer, scheduler = load_model(args)

    model = model.to(args.device)



    with open(f'./ranks/clip/{args.dataset}/wasser_dist_softmaxed.pkl', 'rb') as file:
        probs = pickle.load(file)

    pos_aug = list(probs[args.one_class_idx].keys())[0]
    neg_aug = list(probs[args.one_class_idx].keys())[-1]
    t_pool = [pos_aug, neg_aug]
    aug_list = augl.get_augmentation_list()
    transform_pool = []
    for aug in aug_list:
        for t in t_pool:
            if aug.lower() == t.lower().replace('_', ''):
                transform_pool.append(aug)
    transform_pool = augl.get_augmentation_pool(transform_pool)
    transform_sequence = augl.TransformSequence(transform_pool).to(args.device)
    stats = None 

    train_global_iter = 0
    for epoch in range(0, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        # train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns, colapse_metrics =\
        train_contrastive(stats, model, train_loader, optimizer, transform_sequence, epoch, scheduler, args, train_global_iter, writer)
        
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
        # print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])}")
        if epoch % 10 == 0:
            avg_auc = evaluation(model, args, root_path)
            print(f"Average AUC: {avg_auc}")
            # writer.add_scalar("Eval/avg_auc", avg_auc, epoch)

        if (epoch) % (args.epochs / 10) == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))


        # if np.mean(epoch_loss['loss']) < best_loss:
        #     torch.save(model.state_dict(), os.path.join(save_path, 'best_params.pt'))
        #     best_loss = np.mean(epoch_loss['loss'])

        # last_sim_ps = avg_sim_ps
        # last_sim_ns = avg_sim_ns
        # args.seed += epoch
        # set_seed(args.seed)

    torch.save(model.state_dict(), os.path.join(save_path, 'last_params.pt'))

if __name__ == '__main__':
    main()
