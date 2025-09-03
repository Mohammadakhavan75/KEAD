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

from contrastive import nt_xent

from utils.eval import evaluation
from utils.paths import create_path
from utils.loader import load_model
from utils.parser import args_parser
from utils.simclr_model import SimCLRModel
from dataset_loader import get_loader

# from utils.monitoring import variance_floor


# -------------------------------
# Main training & test routines
# -------------------------------
def train_contrastive(stats, model, classifier, bce_criterion, train_loader, optimizer, pos_transform_layers, neg_transform_layers, epoch, scheduler, args, train_global_iter, writer):
    device = args.device
    losses = {
        'con_loss': [],
        'bc_loss': [],
    }

    # training
    model.train()
    classifier.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for anchor, labels in pbar:
        anchor = anchor.to(device)
        labels = labels.to(device)
        B = anchor.size(0)
        
        # Build two view lists (pre-augmentation) and append their negative pairs
        images1 = anchor.clone()
        images2 = anchor.clone()

        images1 = torch.cat([images1, neg_transform_layers(images1.clone())], dim=0)  # 2B
        images2 = torch.cat([images2, neg_transform_layers(images2.clone())], dim=0)  # 2B

        # Shift labels per snippet: [1s, 0s] then repeat for the second view -> 4B
        shift_labels = torch.cat([torch.ones_like(labels), torch.zeros_like(labels)], dim=0)
        shift_labels = shift_labels.repeat(2).float()  # BCE targets as float

        # Concatenate views and apply SimCLR-style augmentation stochastically to each image
        images_pair = torch.cat([images1, images2], dim=0)  # 4B
        images_pair = pos_transform_layers(images_pair)

        optimizer.zero_grad()

        # One forward over all images; split into two views for NT-Xent
        z_cat, feats_cat = model(images_pair)  # z_cat: (4B, D), feats: (4B, F)

        B2 = 2 * B
        z1 = z_cat[:B2]
        z2 = z_cat[B2:]
        feat_all = feats_cat  # (4B, F)

        # NT-Xent over both original and negative pairs (2B anchors)
        con_loss, sim_p, sim_n, norm_z1, norm_z2 = nt_xent(z1, z2, args.temperature)

        # Binary classification (BCE) with provided shift labels (length 4B)
        logits = classifier(feat_all).view(-1)
        bc_loss = bce_criterion(logits, shift_labels)

        losses['con_loss'].append(con_loss.item())
        losses['bc_loss'].append(bc_loss.item())

        # Final objective: NT-Xent + alpha * BCE (CSI-style)
        loss = con_loss + args.alpha * bc_loss
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
        writer.add_scalar("Train/con_loss", con_loss.item(), train_global_iter)
        writer.add_scalar("Train/bc_loss", bc_loss.item(), train_global_iter)
        writer.add_scalar("Train/sim_p", float(sim_p.item()) if torch.is_tensor(sim_p) else float(sim_p), train_global_iter)
        writer.add_scalar("Train/sim_n", float(sim_n.item()) if torch.is_tensor(sim_n) else float(sim_n), train_global_iter)
        writer.add_scalar("Train/norm_z1", float(norm_z1.item()) if torch.is_tensor(norm_z1) else float(norm_z1), train_global_iter)
        writer.add_scalar("Train/norm_z2", float(norm_z2.item()) if torch.is_tensor(norm_z2) else float(norm_z2), train_global_iter)

        train_global_iter += 1

    return train_global_iter, losses

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

        # Store the arguments in a json file
    config_file_path = os.path.join(args.save_path, 'config_args.json')
    args_dict = vars(args).copy()
    args_dict['device'] = str(args_dict['device'])
    with open(config_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"All config arguments saved to {config_file_path}")

    # This is the less harmful augmentation so the model can learn better.
    general_transform = v2.Compose([
        v2.RandomRotation(degrees=10), # Even if you specify only degrees in RandomAffine, it still goes through the affine pipeline (so performance-wise it’s equivalent but not simpler than RandomRotation)
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.ToTensor(),
    ])

    train_loader, test_loader = get_loader(args, data_path, imagenet_path, general_transform)
    model, optimizer, scheduler = load_model(args)
    # model = SimCLRModel(base_model=args.model)

    model = model.to(args.device)

    # Build a lightweight binary classifier on top of backbone features (BCE: positive views=0, negative views=1)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=args.device)
        _, feat_dim_probe = model(dummy)
        if isinstance(feat_dim_probe, torch.Tensor):
            feat_dim = feat_dim_probe.shape[1]
        else:
            raise RuntimeError("Model did not return features; ensure --proj_head is enabled.")
    model.train()

    classifier = nn.Linear(feat_dim, 1).to(args.device)
    bce_criterion = nn.BCEWithLogitsLoss()
    # Add classifier params to optimizer
    optimizer.add_param_group({'params': classifier.parameters()})

    with open(f'./ranks/clip/{args.dataset}/wasser_dist_softmaxed.pkl', 'rb') as file:
        probs = pickle.load(file)

    sorted_augs = list(probs[args.one_class_idx].keys())

    pos_augs = sorted_augs[:args.n_pos]
    neg_augs = sorted_augs[-args.n_neg:]

    aug_list = augl.get_augmentation_list()
    pos_transform_layers = []
    neg_transform_layers = []
    
    for aug_name in pos_augs:
        for aug in aug_list:
            if aug.lower() == aug_name.lower().replace('_', ''):
                print(f"Using {aug} as positive augmentation")
                pos_transform_layers.append(augl.return_aug(aug, p=0.5).to(args.device))

    for aug_name in neg_augs:
        print(aug_name)
        for aug in aug_list:
            if aug.lower() == aug_name.lower().replace('_', ''):
                print(f"Using {aug} as negative augmentation")
                neg_transform_layers.append(augl.return_aug(aug, p=1.0).to(args.device))


    stats = None


    pos_transform_layers = nn.Sequential(*pos_transform_layers)
    neg_transform_layers = nn.Sequential(*neg_transform_layers)

    train_global_iter = 0
    for epoch in range(0, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        train_global_iter, losses = train_contrastive(stats, model, classifier, bce_criterion, train_loader, optimizer, pos_transform_layers, neg_transform_layers, epoch, scheduler, args, train_global_iter, writer)
        
        scheduler.step()
        args.last_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Train/lr", args.last_lr, epoch)
        writer.add_scalar("Train/con_loss", torch.mean(torch.tensor(losses['con_loss'])), epoch)
        writer.add_scalar("Train/bc_loss", torch.mean(torch.tensor(losses['bc_loss'])), epoch)

        if epoch % 10 == 0:
            avg_auc = evaluation(model, args, root_path)
            print(f"Average AUC: {avg_auc}")
            writer.add_scalar("Eval/avg_auc", avg_auc, epoch)

        if (epoch) % (args.epochs / 100) == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))
            torch.save(classifier.state_dict(), os.path.join(model_save_path, f'classifier_params_epoch_{epoch}.pt'))

    avg_auc = evaluation(model, args, root_path)
    print(f"Average AUC: {avg_auc}")
    writer.add_scalar("Eval/avg_auc", avg_auc, epoch)
    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path, 'last_params.pt'))
    torch.save(classifier.state_dict(), os.path.join(save_path, 'last_classifier.pt'))


if __name__ == '__main__':
    main()
