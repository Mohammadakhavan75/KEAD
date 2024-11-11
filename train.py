import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from datetime import datetime
from contrastive import contrastive, contrastive_matrix
from models.resnet_imagenet import resnet18, resnet50
from models.resnet import ResNet18, ResNet50
from torch.utils.tensorboard import SummaryWriter
from dataset_loader import noise_loader, load_cifar10, load_svhn, load_cifar100, load_imagenet, load_mvtec_ad, load_visa
from sklearn.metrics import accuracy_score

def to_np(x):
    return x.data.cpu().numpy()


def parsing():
    parser = argparse.ArgumentParser(description='',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--from_epoch', type=int, default=1,
                        help='from which epoch start the training.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int,
                        default=0, help='starting epoch from.')
    parser.add_argument('--save_path', type=str,
                        default=None, help='Path to save files.')
    parser.add_argument('--resume', action="store_true",
                        help='Path to model to resume training.')
    parser.add_argument('--device', type=str, 
                        default="cuda", help='cuda or cpu.')
    # Optimizer Config
    parser.add_argument('--optimizer', type=str,
                        default='sgd', help='The initial learning rate.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float,
                        default=3, help='The update rate for learning rate.')
    parser.add_argument('--milestones', nargs='+', type=int, default=[500, 800],
                         help='A list of milestones')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.1, help='The gamma param for updating learning rate.')
    parser.add_argument('--last_lr', type=float,
                        default=0, help='The gamma param for updating learning rate.')
                        
    parser.add_argument('--lamb', type=float, default=1, help='loss scaling parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=1e-6, help='Weight decay (L2 penalty).')

    parser.add_argument('--exp_idx', default=0, type=int, help='run index')

    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10-cifar100-svhn')
    parser.add_argument('--one_class_idx', default=None, type=int, help='select one class index')
    parser.add_argument('--temperature', default=0.5, type=float, help='chaning temperature of contrastive loss')
    parser.add_argument('--preprocessing', default='clip', type=str, help='which preprocessing use for noise order')
    parser.add_argument('--shift_normal', action="store_true", help='Using shifted normal data')
    parser.add_argument('--k_pairs', default=1, type=int, help='Selecting multiple pairs for contrastive loss')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu number')
    parser.add_argument('--multi_gpu', action="store_true", help='Selecting multi-gpu for training')
    parser.add_argument('--e_holder', default=0, type=str, help='Epoch number holder')
    parser.add_argument('--linear', action="store_true", help='Initiate linear layer or not!')
    parser.add_argument('--config', default=None, help='Config file for reading paths')
    parser.add_argument('--img_size', default=32, type=int, help='image size selection')
    parser.add_argument('--model', default='resnet18', type=str, help='resnet model selection')

    parser.add_argument('--l_con', default=1., type=float, help='Weight of Contrastive loss')
    parser.add_argument('--l_bc', default=1., type=float, help='Wieght of BCE loss')
    args = parser.parse_args()

    return args


def train_one_class(train_loader, train_positives_loader, train_negetives_loader,
                     model, train_global_iter, BCELoss, optimizer, args, writer, epoch, scheduler):

    # Enter model into train mode
    model.train()
    
    # Tracking metrics
    epoch_accuracies = {
        'normal': [],
        'positive': [],
        'negative': []
    }
    epoch_loss = {
        'loss': [],
        'ce': [],
        'contrastive': []
    }
    
    sim_ps, sim_ns = [], []
    
    if args.k_pairs == 1:
        for n, (normal, p_data, n_data) in tqdm(enumerate(zip(train_loader, train_positives_loader[0], train_negetives_loader[0]))):
            imgs, labels = normal
            p_imgs, p_label = p_data
            n_imgs, n_label = n_data
            assert torch.equal(torch.squeeze(labels), torch.squeeze(p_label)), f"The labels of positives images do not match to noraml images, {torch.squeeze(labels)}, {torch.squeeze(p_label)}"
            assert torch.equal(torch.squeeze(labels), torch.squeeze(n_label)), f"The labels of negatives images do not match to noraml images, {torch.squeeze(labels)}, {torch.squeeze(n_label)}"
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            p_imgs = p_imgs.to(args.device)
            n_imgs = n_imgs.to(args.device)
        
            optimizer.zero_grad()
            pred_d, normal_features = model(imgs, True)
            pred_p, p_features = model(p_imgs, True)
            pred_n, n_features = model(n_imgs, True)
            normal_features = normal_features[-1]
            p_features = p_features[-1]
            n_features = n_features[-1]

            loss_contrastive = torch.tensor(0.0, requires_grad=True)
            loss_contrastive, sim_p, sim_n, data_norm, negative_norms, positive_norms = contrastive_matrix(normal_features, p_features, n_features, temperature=args.temperature)
            sim_ps.append(torch.mean(sim_p).detach().cpu())
            sim_ns.append(torch.mean(sim_n).detach().cpu())

            labels = torch.zeros(len(pred_d)*3).to(args.device)
            labels[len(pred_d)*2:] = 1.
            preds = torch.cat((pred_d, pred_p, pred_n), dim=0).squeeze(1).to(args.device)
 
            bc_loss = BCELoss(preds, labels)
            loss = args.l_con * loss_contrastive + args.l_bc * bc_loss
            
            epoch_loss['loss'].append(loss.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            train_global_iter += 1
            writer.add_scalar("Train/loss_contrastive", loss_contrastive.item(), train_global_iter)
            writer.add_scalar("Train/bc_loss", bc_loss.item(), train_global_iter)
            writer.add_scalar("Train/loss", loss.item(), train_global_iter)
            writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/norm_p", torch.mean(data_norm).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/norm_n", torch.mean(negative_norms).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/norm_d", torch.mean(positive_norms).detach().cpu().numpy(), train_global_iter)


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
            scheduler.step(epoch - 1 + n / len(train_loader))
            args.last_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Train/lr", args.last_lr, epoch)

    elif args.k_pairs == 2:
        for n, (normal, p1_data, n1_data, p2_data, n2_data) in tqdm(enumerate(zip(train_loader, train_positives_loader[0], train_negetives_loader[0], train_positives_loader[1], train_negetives_loader[1]))):
            imgs, labels = normal
            p1_imgs, p1_label = p1_data
            n1_imgs, n1_label = n1_data
            p2_imgs, p2_label = p2_data
            n2_imgs, n2_label = n2_data
            
            assert torch.equal(torch.squeeze(labels), torch.squeeze(p1_label)), f"The labels of positives 1 images do not match to noraml images, {torch.squeeze(labels)}, {torch.squeeze(p1_label)}"
            assert torch.equal(torch.squeeze(labels), torch.squeeze(n1_label)), f"The labels of negatives 1 images do not match to noraml images, {torch.squeeze(labels)}, {torch.squeeze(n1_label)}"
            assert torch.equal(torch.squeeze(labels), torch.squeeze(p2_label)), f"The labels of positives 2 images do not match to noraml images, {torch.squeeze(labels)}, {torch.squeeze(p2_label)}"
            assert torch.equal(torch.squeeze(labels), torch.squeeze(n2_label)), f"The labels of negatives 2 images do not match to noraml images, {torch.squeeze(labels)}, {torch.squeeze(n2_label)}"

            imgs, labels = imgs.to(args.device), labels.to(args.device)
            p1_imgs = p1_imgs.to(args.device)
            n1_imgs = n1_imgs.to(args.device)
            p2_imgs = p2_imgs.to(args.device)
            n2_imgs = n2_imgs.to(args.device)
        
        
            optimizer.zero_grad()
            pred_d, normal_features = model(imgs, True)
            p1_pred, p1_features = model(p1_imgs, True)
            n1_pred, n1_features = model(n1_imgs, True)
            p2_pred, p2_features = model(p2_imgs, True)
            n2_pred, n2_features = model(n2_imgs, True)

            normal_features = normal_features[-1]
            p1_features = p1_features[-1]
            n1_features = n1_features[-1]
            p2_features = p2_features[-1]
            n2_features = n2_features[-1]

            loss_contrastive = torch.tensor(0.0, requires_grad=True)

            p_features = torch.cat([p1_features, p2_features], dim=0)
            n_features = torch.cat([n1_features, n2_features], dim=0)
            loss_contrastive, sim_p, sim_n, data_norm, negative_norms, positive_norms = contrastive_matrix(normal_features, p_features, n_features, temperature=args.temperature)

            sim_ps.append(torch.mean(sim_p).detach().cpu())
            sim_ns.append(torch.mean(sim_n).detach().cpu())

            labels = torch.zeros(len(pred_d) * 5).to(args.device)
            labels[len(pred_d) * 3:] = 1.
            preds = torch.cat((pred_d, p1_pred, p2_pred, n1_pred, n2_pred), dim=0).squeeze(1).to(args.device)

            bc_loss = BCELoss(preds, labels)
            loss = loss_contrastive + bc_loss
            
            epoch_loss['loss'].append(loss.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            writer.add_scalar("Train/loss_contrastive", loss_contrastive.item(), train_global_iter)
            writer.add_scalar("Train/bc_loss", bc_loss.item(), train_global_iter)
            writer.add_scalar("Train/loss", loss.item(), train_global_iter)
            writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/norm_p", torch.mean(data_norm).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/norm_n", torch.mean(negative_norms).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/norm_d", torch.mean(positive_norms).detach().cpu().numpy(), train_global_iter)
            
            loss.backward()
            pp = []
            for param in model.parameters():
                if param.grad is not None:
                    pp.append(torch.mean(param.grad.norm()).detach().cpu())
            
            writer.add_scalar("Train/grads", torch.mean(torch.tensor(pp)).detach().cpu().numpy(), train_global_iter)

            optimizer.step()
            scheduler.step(epoch - 1 + n / len(train_loader))
            args.last_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Train/lr", args.last_lr, epoch)

    elif args.k_pairs == 3:
        for normal, p_data, n_data, p_data1, n_data1, p_data2, n_data2 in zip(train_loader, train_positives_loader[0], train_negetives_loader[0], train_positives_loader[1], train_negetives_loader[1], train_positives_loader[2], train_negetives_loader[2]):
            imgs, labels = normal
            p_imgs, _ = p_data
            n_imgs, _ = n_data
            p_imgs1, _ = p_data1
            n_imgs1, _ = n_data1
            p_imgs2, _ = p_data2
            n_imgs2, _ = n_data2
            
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            p_imgs = p_imgs.to(args.device)
            n_imgs = n_imgs.to(args.device)
            p_imgs1 = p_imgs1.to(args.device)
            n_imgs1 = n_imgs1.to(args.device)
            p_imgs2 = p_imgs2.to(args.device)
            n_imgs2 = n_imgs2.to(args.device)
        
        
            optimizer.zero_grad()
            _, normal_features = model(imgs, True)
            _, p_features = model(p_imgs, True)
            _, n_features = model(n_imgs, True)
            _, p_features1 = model(p_imgs1, True)
            _, n_features1 = model(n_imgs1, True)
            _, p_features2 = model(p_imgs2, True)
            _, n_features2 = model(n_imgs2, True)

            normal_features = normal_features[-1]
            p_features = p_features[-1]
            n_features = n_features[-1]
            p_features1 = p_features1[-1]
            n_features1 = n_features1[-1]
            p_features2 = p_features2[-1]
            n_features2 = n_features2[-1]

            loss_contrastive = torch.tensor(0.0, requires_grad=True)

            for norm_f, p_f, n_f, p_f1, n_f1, p_f2, n_f2  in zip(normal_features, p_features, n_features, p_features1, n_features1, p_features2, n_features2):
                p_fs = torch.stack([p_f, p_f1, p_f2], dim=0)
                n_fs = torch.stack([n_f, n_f1, n_f2], dim=0)
                l_c, sim_p, sim_n = contrastive(norm_f, p_fs, n_fs, temperature=args.temperature)
                loss_contrastive = loss_contrastive + l_c
                sim_ps.append(sim_p.detach().cpu())
                sim_ns.append(sim_n.detach().cpu())
            
        
            loss = loss_contrastive / len(normal_features)
            
            epoch_loss['loss'].append(loss.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            train_global_iter += 1
            writer.add_scalar("Train/loss", loss.item(), train_global_iter)
            writer.add_scalar("Train/sim_p", torch.mean(sim_p).detach().cpu().numpy(), train_global_iter)
            writer.add_scalar("Train/sim_n", torch.mean(sim_n).detach().cpu().numpy(), train_global_iter)

            loss.backward()
            optimizer.step()
    
    if args.k_pairs == 1:
        avg_sim_ps = torch.mean(torch.tensor(sim_ps), dim=0).detach().cpu().numpy()
        avg_sim_ns = torch.mean(torch.tensor(sim_ns), dim=0).detach().cpu().numpy()
        writer.add_scalar("AVG_Train/sim_p", avg_sim_ps, train_global_iter)
        writer.add_scalar("AVG_Train/sim_n", avg_sim_ns, train_global_iter)

    else:
        avg_sim_ps = torch.mean(torch.tensor(sim_ps), dim=0).detach().cpu().numpy()
        avg_sim_ns = torch.mean(torch.tensor(sim_ns), dim=0).detach().cpu().numpy()
        writer.add_scalar("AVG_Train/sim_p", avg_sim_ps, train_global_iter)
        writer.add_scalar("AVG_Train/sim_n", avg_sim_ns, train_global_iter)

    return train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns


def calculate_accuracy(model_output, true_labels):
    # Convert sigmoid output to class predictions
    predicted_classes = [1 if x > 0.5 else 0 for x in model_output]
    
    # Flatten the tensor if necessary
    if len(predicted_classes) != len(true_labels):
        predicted_classes = torch.tensor(predicted_classes).view(-1)
        
    accuracy = accuracy_score(true_labels, predicted_classes)
    return accuracy


def evaluate(eval_in, net, args):
    net = net.to(args.device)
    net.eval()  # enter valid mode

    model_preds = []
    trues = []
    with torch.no_grad():
        for inputs_in, _ in eval_in:
            inputs_in = inputs_in.to(args.device)

            preds_in, _ = net(inputs_in, True)

            model_preds.extend(preds_in.detach().cpu().numpy())
            trues.extend(torch.zeros_like(preds_in).detach().cpu().numpy())
            
        accuracy = calculate_accuracy(model_preds, trues)
    return accuracy

def load_model(args):

    if args.img_size == 224:
        if args.model == 'resnet18':
            model = resnet18(num_classes=args.num_classes)
        elif args.model == 'resnet50':
            model = resnet50(num_classes=args.num_classes)
        else:
            raise NotImplementedError("Not implemented model!")
        
    elif args.img_size == 32:
        if args.linear:
            model = ResNet18(args.num_classes)
        else:
            if args.model == 'resnet18':
                model = ResNet18(num_classes=args.num_classes)
            elif args.model == 'resnet50':
                model = ResNet50(num_classes=args.num_classes)
            else:
                raise NotImplementedError("Not implemented model!")
    else:
        raise NotImplementedError("Not implemented image size!")
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                momentum=args.momentum,weight_decay=args.decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                                    weight_decay=args.decay)
    else:
        raise NotImplementedError("Not implemented optimizer!")

    if args.resume:
        model_folder = args.save_path + 'models/'
        model_name = natsorted(os.listdir(model_folder))[-1]
        args.from_epoch = int(model_name.split('_')[-1].split('.')[0])
        model.load_state_dict(torch.load(os.path.join(model_folder, model_name), weights_only=True))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    BCELoss = torch.nn.BCEWithLogitsLoss().to(args.device)

    return model, BCELoss, optimizer, scheduler


def create_path(args):
    if args.resume:
        print(args.save_path)
        print(args.save_path is None)
        assert args.save_path is not None, "You must pass the model path in resume mode!"
        save_path = args.save_path
        model_save_path = save_path + 'models/'
    else:
        root_addr = f'./run/exp_{args.exp_idx}/'
        if args.exp_idx == 0:
            while os.path.exists(root_addr):
                args.exp_idx += 1
                root_addr = root_addr.split('_')[0] + '_' + str(args.exp_idx) + '/'
            
            os.makedirs(root_addr)


        save_path = root_addr + f'{args.dataset}' + f'_one_class_idx_{args.one_class_idx}' + f'_preprocessing_{args.preprocessing}' + \
                    f'_{args.model}' + f'_{args.img_size}' + f'_{args.optimizer}' + f'_lr_{args.learning_rate}' + \
                    f'_lru_{args.lr_update_rate}' + f'_lrg_{args.lr_gamma}' + f'_epochs_{args.epochs}'  + \
                    f'_temprature_{args.temperature}' + f'_seed_{args.seed}' + f'_k_pairs_{args.k_pairs}' + '/'
        
        model_save_path = save_path + 'models/'
        os.makedirs(model_save_path, exist_ok=True)

    return model_save_path, save_path


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
    
    print("Start Loading noises")
    train_positives_loader, train_negetives_loader,\
    test_positives_loader, test_negetives_loader = noise_loader(
                    args,
                    batch_size=args.batch_size,
                    one_class_idx=args.one_class_idx,
                    resize=args.img_size,
                    dataset=args.dataset,
                    preprocessing=args.preprocessing, 
                    k_pairs=args.k_pairs,
                    seed=args.seed)
    print("Loading noises finished!")

    return train_loader, test_loader, train_positives_loader, train_negetives_loader, test_positives_loader, test_negetives_loader


def set_seed(seed_nu):
    torch.manual_seed(seed_nu)
    random.seed(seed_nu)
    np.random.seed(seed_nu)


if __name__ == "__main__":

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    root_path = config['root_path']
    data_path = config['data_path']
    imagenet_path = config['imagenet_path']
    args = parsing()
    args.config = config
    
    best_loss = torch.inf
    # Setting seeds for reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_save_path, save_path = create_path(args)
    writer = SummaryWriter(save_path)
    args.save_path = save_path

    train_loader, test_loader, train_positives_loader, train_negetives_loader,\
        test_positives_loader, test_negetives_loader = loading_datasets(args, data_path, imagenet_path)

    model, BCELoss, optimizer, scheduler = load_model(args)

    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    elif args.device == 'cuda':    
        os.environ['CUDA_VISIBLE_DEVICES']='0,1'
        args.device=f'cuda:{args.gpu}'

    model = model.to(args.device)
    train_global_iter = 0

    args.last_lr = args.learning_rate
    for epoch in range(args.from_epoch, args.epochs):
        print('epoch', epoch, '/', args.epochs)
        args.e_holder = str(epoch)
        train_global_iter, epoch_loss, epoch_accuracies, avg_sim_ps, avg_sim_ns =\
            train_one_class(train_loader, train_positives_loader, train_negetives_loader, \
                            model, train_global_iter, BCELoss, optimizer, args, writer, epoch, scheduler)
        
        writer.add_scalar("AVG_Train/avg_loss", np.mean(epoch_loss['loss']), epoch)
        writer.add_scalar("AVG_Train/avg_loss_contrastive", np.mean(epoch_loss['contrastive']), epoch)
        writer.add_scalar("Train/lr", args.last_lr, epoch)

        accuracy = evaluate(test_loader, model, args)
        print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])}, Train/avg_acc: {accuracy}")
        
        if (epoch) % (args.epochs / 10) == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_params_epoch_{epoch}.pt'))


        if np.mean(epoch_loss['loss']) < best_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'best_params.pt'))
            best_loss = np.mean(epoch_loss['loss'])

        last_sim_ps = avg_sim_ps
        last_sim_ns = avg_sim_ns
        args.seed += epoch
        set_seed(args.seed)

    torch.save(model.state_dict(), os.path.join(save_path, 'last_params.pt'))
