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

from utils.monitoring import variance_floor


# -------------------------------
# Main training & test routines
# -------------------------------
def train_contrastive(stats, model, train_loader, optimizer, pos_transform_layers, neg_transform_layers, epoch, scheduler, args, train_global_iter, writer):
    device = args.device
    losses = {
        'var_loss': [],
        'con_loss': [],
    }
    n_pos = len(pos_transform_layers)

    # training
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for anchor, _ in pbar:
        anchor = anchor.to(device)
        B = anchor.size(0)
        
        pos_views = torch.cat([pos_transform_layer(anchor) for pos_transform_layer in pos_transform_layers], dim=0)
        if not args.seq_aug:
            neg_views = torch.cat([neg_transform_layer(anchor) for neg_transform_layer in neg_transform_layers], dim=0)
        else:
            neg_views = neg_transform_layers(anchor)

        
        optimizer.zero_grad()
        
        model_input = torch.cat([anchor, pos_views, neg_views], dim=0)
        preds, feats = model(model_input)
        
        rep_a = preds[:B]
        rep_p = preds[B:B * (1 + n_pos)]
        rep_n = preds[B * (1 + n_pos):]
        
        feats_a = feats[:B]
        feats_p = feats[B:B * (1 + n_pos)]
        
        con_loss, sim_p, sim_n, norm_a, norm_n, norm_p = contrastive_matrix(
            rep_a, rep_p, rep_n, args.temperature, positives_per_anchor=n_pos
        )

        # Applying the variance floor on backbone output instead of proj head.
        # So the head can then focus on alignment but the cloud volume is guaranteed upstream.
        feat_align = torch.cat([rep_a, rep_p], dim=0)
        var_loss, std_dev = variance_floor(feat_align, gamma=1.0)

        losses['con_loss'].append(con_loss.item())
        losses['var_loss'].append(var_loss.item())

        loss = con_loss + var_loss
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
        writer.add_scalar("Train/std_min", std_dev.min().item(), train_global_iter)
        writer.add_scalar("Train/std_max", std_dev.max().item(), train_global_iter)

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


    transform = v2.Compose([
                # Geometric (light): keep horizon horizontal
                v2.RandomResizedCrop(args.img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=5, translate=(0.10, 0.10), shear=5),
                # Photometric
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # Tensor + cutout last
                
                v2.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
                v2.ToTensor(),
                # (optional) Normalize with CIFAR-10 mean/std
                # v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    train_loader, test_loader = get_loader(args, data_path, imagenet_path, transform)
    model, optimizer, scheduler = load_model(args)
    # model = SimCLRModel(base_model=args.model)

    model = model.to(args.device)

    with open(f'./ranks/clip/{args.dataset}/wasser_dist_softmaxed.pkl', 'rb') as file:
        probs = pickle.load(file)

    sorted_augs = list(probs[args.one_class_idx].keys())
    pos_augs = [aug for aug in sorted_augs[:args.n_pos]]
    neg_augs = [aug for aug in sorted_augs[-args.n_neg:]]


    aug_list = augl.get_augmentation_list()
    pos_transform_layers = []
    neg_transform_layers = []
    
    for aug_name in pos_augs:
        for aug in aug_list:
            if aug.lower() == aug_name.lower().replace('_', ''):
                print(f"Using {aug} as positive augmentation")
                pos_transform_layers.append(augl.return_aug(aug).to(args.device))

    for aug_name in neg_augs:
        for aug in aug_list:
            if aug.lower() == aug_name.lower().replace('_', ''):
                print(f"Using {aug} as negative augmentation")
                neg_transform_layers.append(augl.return_aug(aug).to(args.device))


    assert len(pos_transform_layers) == args.n_pos, f"Expected {args.n_pos} positive augmentations, but found {len(pos_transform_layers)}"
    assert len(neg_transform_layers) == args.n_neg, f"Expected {args.n_neg} negative augmentations, but found {len(neg_transform_layers)}"

    stats = None

    if args.seq_aug:
        neg_transform_layers = nn.Sequential(*neg_transform_layers)

    train_global_iter = 0
    for epoch in range(0, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        train_global_iter, losses = train_contrastive(stats, model, train_loader, optimizer, pos_transform_layers, neg_transform_layers, epoch, scheduler, args, train_global_iter, writer)
        
        scheduler.step()
        args.last_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Train/lr", args.last_lr, epoch)
        writer.add_scalar("Train/con_loss", torch.mean(torch.tensor(losses['con_loss'])), epoch)
        writer.add_scalar("Train/var_loss", torch.mean(torch.tensor(losses['var_loss'])), epoch)

        if epoch % 10 == 0:
            avg_auc = evaluation(model, args, root_path)
            print(f"Average AUC: {avg_auc}")
            writer.add_scalar("Eval/avg_auc", avg_auc, epoch)

        if (epoch) % (args.epochs / 100) == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))

    avg_auc = evaluation(model, args, root_path)
    print(f"Average AUC: {avg_auc}")
    writer.add_scalar("Eval/avg_auc", avg_auc, epoch)
    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path, 'last_params.pt'))


if __name__ == '__main__':
    main()