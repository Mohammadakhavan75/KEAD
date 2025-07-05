import os
import json
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.utils import augmentation_layers as augl
import torchvision.transforms.v2 as v2

from contrastive import contrastive_matrix

from utils.eval import evaluation
from utils.paths import create_path
from utils.loader import load_model
from utils.parser import args_parser
from utils.simclr_model import SimCLRModel
from dataset_loader import get_loader


# -------------------------------
# Main training & test routines
# -------------------------------
def train_contrastive(stats, model, train_loader, optimizer, pos_transform_layer, neg_transform_layer, epoch, scheduler, args, train_global_iter, writer):
    device = args.device

    # training
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for anchor, _ in pbar:
        anchor = anchor.to(device)
        B = anchor.size(0)
        
        pos_views = pos_transform_layer(anchor)
        neg_views = neg_transform_layer(anchor)
        optimizer.zero_grad()
        
        model_input = torch.cat([anchor, pos_views, neg_views], dim=0)
        preds, reps_list = model(model_input)
        rep_a, rep_p, rep_n = preds[:B], preds[B:2*B], preds[2*B:]
        loss, sim_p, sim_n, norm_a, norm_n, norm_p = contrastive_matrix(rep_a, rep_p, rep_n, args.temperature)

        loss.backward()
        pp = []
        for param in model.parameters():
            if param.grad is not None:
                pp.append(torch.mean(param.grad.norm()).detach().cpu())
        pd = []
        for param in model.parameters():
            pd.append(torch.mean(param.data.norm()).detach().cpu())
        
        writer.add_scalar("Train/params", torch.mean(torch.tensor(pd)).detach().cpu().numpy(), train_global_iter)
        writer.add_scalar("Train/grads", torch.mean(torch.tensor(pp)).detach().cpu().numpy(), train_global_iter)


        optimizer.step()
        pbar.set_postfix(loss=loss.item())

        writer.add_scalar("Train/loss", loss.item(), train_global_iter)
        writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
        writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)
        writer.add_scalar("Train/norm_a", torch.mean(norm_a).detach().cpu().numpy(), train_global_iter)
        writer.add_scalar("Train/norm_n", torch.mean(norm_n).detach().cpu().numpy(), train_global_iter)
        writer.add_scalar("Train/norm_p", torch.mean(norm_p).detach().cpu().numpy(), train_global_iter)


        scheduler.step()
        args.last_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Train/lr", args.last_lr, epoch)

        train_global_iter += 1

    return train_global_iter

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
    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device(args.device)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    model = SimCLRModel(base_model=args.model)

    model = model.to(args.device)

    with open(f'./ranks/clip/{args.dataset}/wasser_dist_softmaxed.pkl', 'rb') as file:
        probs = pickle.load(file)

    pos_aug = list(probs[args.one_class_idx].keys())[0]
    neg_aug = list(probs[args.one_class_idx].keys())[-1]
    
    aug_list = augl.get_augmentation_list()
    pos_transform_layer = None
    neg_transform_layer = None
    
    for aug in aug_list:
        if aug.lower() == pos_aug.lower().replace('_', ''):
            pos_transform_layer = augl.return_aug(aug).to(args.device)
        elif aug.lower() == neg_aug.lower().replace('_', ''):
            neg_transform_layer = augl.return_aug(aug).to(args.device)

    assert pos_transform_layer is not None, "pos_transform_layer is None"
    assert neg_transform_layer is not None, "neg_transform_layer is None"

    stats = None

    train_global_iter = 0
    for epoch in range(0, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        # train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns, colapse_metrics =\
        train_global_iter = train_contrastive(stats, model, train_loader, optimizer, pos_transform_layer, neg_transform_layer, epoch, scheduler, args, train_global_iter, writer)
        
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
