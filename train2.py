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
from utils.learnable_policy import LearnableAugPolicy

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
def train_contrastive(stats, model, train_loader, optimizer, pos_transform_layers, neg_transform_layers, epoch, scheduler, args, train_global_iter, writer, policy: LearnableAugPolicy | None = None, policy_optim = None):
    device = args.device
    losses = {
        'var_loss': [],
        'con_loss': [],
    }
    n_pos = len(pos_transform_layers) if policy is None else args.n_pos
    # Track per-epoch selection counts when using policy
    pos_select_counts = {}
    neg_select_counts = {}
    if policy is not None:
        for nm in policy.aug_names:
            pos_select_counts[nm] = 0
            neg_select_counts[nm] = 0

    # training
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for anchor, _ in pbar:
        anchor = anchor.to(device)
        B = anchor.size(0)
        if policy is None:
            pos_views = torch.cat([pos_transform_layer(anchor) for pos_transform_layer in pos_transform_layers], dim=0)
            if not args.seq_aug:
                neg_views = torch.cat([neg_transform_layer(anchor) for neg_transform_layer in neg_transform_layers], dim=0)
            else:
                neg_views = neg_transform_layers(anchor)
            logp_pos = None
            logp_neg = None
            pos_idx = None
            neg_idx = None
        else:
            policy_optim.zero_grad()
            pos_views, neg_views, logp_pos, logp_neg, pos_idx, neg_idx = policy(anchor, n_pos=args.n_pos, n_neg=args.n_neg)
            # Record which augmentations were selected this step
            pnames, nnames = policy.selected_names(pos_idx, neg_idx)
            for nm in pnames:
                pos_select_counts[nm] += 1
            for nm in nnames:
                neg_select_counts[nm] += 1
            # Add a short text log to TensorBoard for quick inspection
            writer.add_text("Policy/selected_pos", ", ".join(pnames), train_global_iter)
            writer.add_text("Policy/selected_neg", ", ".join(nnames), train_global_iter)

        
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
        # feat_align = torch.cat([rep_a, rep_p], dim=0)
        feat_align = torch.cat([feats_a, feats_p], dim=0)
        var_loss, std_dev = variance_floor(feat_align, gamma=1.0)

        losses['con_loss'].append(con_loss.item())
        losses['var_loss'].append(var_loss.item())

        loss = con_loss + var_loss

        # Policy learning: REINFORCE + entropy regularization
        if policy is not None:
            # Update EMA baseline first
            policy.update_baseline(loss.detach(), momentum=args.policy_baseline_momentum)

            # Advantage: smaller is better (we minimize loss)
            advantage = (loss.detach() - policy.baseline)

            # For adversarial negative policy, flip sign (hard-negative mining)
            if args.policy_neg_adversarial and logp_neg is not None:
                policy_loss = args.policy_coef * (advantage * (logp_pos - logp_neg))
            else:
                # Cooperative: both pos and neg chosen to minimize training loss
                # Note: no gradient flows into encoder from these terms; only into policy logits
                logp_total = 0.0
                if logp_pos is not None:
                    logp_total = logp_total + logp_pos
                if logp_neg is not None:
                    logp_total = logp_total + logp_neg
                policy_loss = args.policy_coef * (advantage * logp_total)

            # Entropy regularization to avoid collapse to trivial transforms
            H_pos, H_neg = policy.entropy()
            entropy_reg = -args.policy_entropy_coef * (H_pos + H_neg)
            policy_loss = policy_loss + entropy_reg

            policy_loss.backward()
            policy_optim.step()

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
        if policy is not None:
            writer.add_scalar("Policy/entropy_pos", policy.entropy()[0].item(), train_global_iter)
            writer.add_scalar("Policy/entropy_neg", policy.entropy()[1].item(), train_global_iter)
            writer.add_scalar("Policy/baseline", policy.baseline.item(), train_global_iter)
            if logp_pos is not None:
                writer.add_scalar("Policy/logp_pos", logp_pos.item(), train_global_iter)
            if logp_neg is not None:
                writer.add_scalar("Policy/logp_neg", logp_neg.item(), train_global_iter)

        train_global_iter += 1

    return train_global_iter, losses, pos_select_counts, neg_select_counts

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
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=5, translate=(0.10, 0.10), shear=5),
                v2.RandomApply([v2.ColorJitter(0.05, 0.05, 0.05, 0.01)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.ToTensor(),
                # (optional) Normalize with CIFAR-10 mean/std
                # v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    train_loader, test_loader = get_loader(args, data_path, imagenet_path, transform)
    model, optimizer, scheduler = load_model(args)
    # model = SimCLRModel(base_model=args.model)

    model = model.to(args.device)

    # Build augmentation selection: fixed or learnable policy
    # Candidate list comes from CLIP-ranks file if present; fallback to all known augs
    with open(f'./ranks/clip/{args.dataset}/wasser_dist_softmaxed.pkl', 'rb') as file:
        probs = pickle.load(file)
    sorted_augs = list(probs[args.one_class_idx].keys())

    policy_module = None
    policy_optim = None
    pos_transform_layers = []
    neg_transform_layers = []

    if args.learnable_policy:
        # Use the entire ranked list as candidate pool; the policy learns pos/neg allocations
        policy_module = LearnableAugPolicy(sorted_augs, severity=args.aug_severity, device=args.device)
        # Add policy params to optimizer
        policy_lr = args.policy_lr if args.policy_lr is not None else args.learning_rate
        policy_optim = torch.optim.Adam(policy_module.parameters(), lr=policy_lr)
        print(f"Initialized learnable policy over {len(sorted_augs)} augmentations (lr={policy_lr}).")
    else:
        # Fixed selection from head/tail of ranked list
        pos_augs = sorted_augs[:args.n_pos]
        neg_augs = sorted_augs[-args.n_neg:]

        aug_list = augl.get_augmentation_list()

        for aug_name in pos_augs:
            for aug in aug_list:
                if aug.lower() == aug_name.lower().replace('_', ''):
                    print(f"Using {aug} as positive augmentation")
                    pos_transform_layers.append(augl.return_aug(aug, severity=args.aug_severity).to(args.device))

        for aug_name in neg_augs:
            for aug in aug_list:
                if aug.lower() == aug_name.lower().replace('_', ''):
                    print(f"Using {aug} as negative augmentation")
                    neg_transform_layers.append(augl.return_aug(aug, severity=args.aug_severity).to(args.device))

        assert len(pos_transform_layers) == args.n_pos, f"Expected {args.n_pos} positive augmentations, but found {len(pos_transform_layers)}"
        assert len(neg_transform_layers) == args.n_neg, f"Expected {args.n_neg} negative augmentations, but found {len(neg_transform_layers)}"

    stats = None

    if (not args.learnable_policy) and args.seq_aug:
        neg_transform_layers = nn.Sequential(*neg_transform_layers)

    train_global_iter = 0
    for epoch in range(0, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        train_global_iter, losses, pos_counts_epoch, neg_counts_epoch = train_contrastive(
            stats, model, train_loader, optimizer,
            pos_transform_layers, neg_transform_layers,
            epoch, scheduler, args, train_global_iter, writer,
            policy_module, policy_optim,
        )
        
        scheduler.step()
        args.last_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Train/lr", args.last_lr, epoch)
        writer.add_scalar("Train/con_loss", torch.mean(torch.tensor(losses['con_loss'])), epoch)
        writer.add_scalar("Train/var_loss", torch.mean(torch.tensor(losses['var_loss'])), epoch)

        # Per-epoch policy diagnostics (counts and current probs)
        if policy_module is not None:
            # Selection counts per augmentation this epoch
            for nm, c in pos_counts_epoch.items():
                writer.add_scalar(f"Policy/pos_count/{nm}", c, epoch)
            for nm, c in neg_counts_epoch.items():
                writer.add_scalar(f"Policy/neg_count/{nm}", c, epoch)

            # Current probability distributions per augmentation
            with torch.no_grad():
                prob_pos = torch.softmax(policy_module.pos_logits, dim=0).detach().cpu()
                prob_neg = torch.softmax(policy_module.neg_logits, dim=0).detach().cpu()
                for i, nm in enumerate(policy_module.aug_names):
                    writer.add_scalar(f"Policy/prob_pos/{nm}", float(prob_pos[i].item()), epoch)
                    writer.add_scalar(f"Policy/prob_neg/{nm}", float(prob_neg[i].item()), epoch)

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

    # Save final policy probabilities table
    if policy_module is not None:
        import csv
        with torch.no_grad():
            p_pos = torch.softmax(policy_module.pos_logits, dim=0).detach().cpu().tolist()
            p_neg = torch.softmax(policy_module.neg_logits, dim=0).detach().cpu().tolist()
            out_csv = os.path.join(save_path, 'policy_final_probs.csv')
            with open(out_csv, 'w', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(['augmentation', 'pos_prob', 'neg_prob'])
                for nm, pp, pn in zip(policy_module.aug_names, p_pos, p_neg):
                    writer_csv.writerow([nm, f"{pp:.6f}", f"{pn:.6f}"])
            print(f"Saved final policy probabilities to {out_csv}")


if __name__ == '__main__':
    main()
