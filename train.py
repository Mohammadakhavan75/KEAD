import os
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from contrastive import contrastive
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from cifar10.model import ResNet18, ResNet34, ResNet50
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score

def to_np(x):
    return x.data.cpu().numpy()

def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='starting epoch from.')
    parser.add_argument('--start_epoch', type=int, 
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
                        default='sgd', help='The initial learning rate.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float,
                        default=5, help='The update rate for learning rate.')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.9, help='The gamma param for updating learning rate.')
    parser.add_argument('--last_lr', type=float,
                        default=0, help='The gamma param for updating learning rate.')
                        
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')
    
    parser.add_argument('--noise', default='gaussian_noise', type=str, help='which noise to be run')
    parser.add_argument('--run_index', default=0, type=int, help='run index')
    
    args = parser.parse_args()

    return args


def train(train_loader, positives, negetives, net, train_global_iter, criterion, optimizer, device, writer):

    print("Traning...")
    net = net.to(device)
    net.train()  # enter train mode

    # track train classification accuracy
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
    for normal, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 in zip(train_loader,
                       positives[0], positives[1], positives[2], positives[3], positives[4],
                       positives[5], positives[6], positives[7], positives[8], positives[9],
                       negetives[0], negetives[1], negetives[2], negetives[3], negetives[4],
                       negetives[5], negetives[6], negetives[7], negetives[8], negetives[9]):
        ps = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]]
        ns = [n0[0], n1[0], n2[0], n3[0], n4[0], n5[0], n6[0], n7[0], n8[0], n9[0]]
        
        imgs, labels = normal
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        positive_imgs = torch.zeros_like(imgs)
        negative_imgs = torch.zeros_like(imgs)
        for i, label in enumerate(labels):
            positive_imgs[i] = ps[label][i].to(args.device)
            negative_imgs[i] = ns[label][i].to(args.device)
            # positive_imgs, negative_imgs = np.asarray(positive_imgs), np.asarray(negative_imgs)

        optimizer.zero_grad()
        preds, normal_features = model(imgs, True)
        positive_preds, positive_features = model(positive_imgs, True)
        negative_preds, negative_features = model(negative_imgs, True)

        normal_probs = torch.softmax(preds, dim=1)
        positive_probs = torch.softmax(positive_preds, dim=1)
        negative_probs = torch.softmax(negative_preds, dim=1)

        # Calculate loss contrastive for layer
        loss_contrastive = 0
        for norm_f, pos_f, neg_f in zip(normal_features[-1], positive_features[-1], negative_features[-1]):
            loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, neg_f))

        loss_ce = criterion(preds, labels)
        loss = loss_ce + loss_contrastive

        normal_output_index = torch.argmax(normal_probs, dim=1)
        positive_output_index = torch.argmax(positive_probs, dim=1)
        negative_output_index = torch.argmax(negative_probs, dim=1)
        acc_normal = accuracy_score(list(to_np(normal_output_index)), list(to_np(labels)))
        acc_positive = accuracy_score(list(to_np(positive_output_index)), list(to_np(labels)))
        acc_negative = accuracy_score(list(to_np(negative_output_index)), list(to_np(labels)))

        # Logging section
        epoch_accuracies['normal'].append(acc_normal)
        epoch_accuracies['positive'].append(acc_positive)
        epoch_accuracies['negative'].append(acc_negative)
        epoch_loss['loss'].append(loss)
        epoch_loss['ce'].append(loss_ce)
        epoch_loss['contrastive'].append(loss_contrastive)

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss, train_global_iter)
        writer.add_scalar("Train/loss_ce", loss_ce.item(), train_global_iter)
        writer.add_scalar("Train/loss_contrastive", loss_contrastive, train_global_iter)
        writer.add_scalar("Train/acc_normal", acc_normal, train_global_iter)
        writer.add_scalar("Train/acc_positive", acc_positive, train_global_iter)
        writer.add_scalar("Train/acc_negative", acc_negative, train_global_iter)

        loss.backward()
        optimizer.step()

    return train_global_iter, epoch_loss, epoch_accuracies 


def test(loader, net, global_eval_iter, criterion, device, writer):

    print("Evaluating...")
    net = net.to(device)
    net.eval()  # enter train mode

    # track train classification accuracy
    eval_acc = []
    eval_loss = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            
            inputs , targets = inputs.to(device) , targets.to(device)

            preds = net(inputs)
            probs = torch.softmax(preds, dim=1)

            loss = criterion(preds, targets)
            
            output_index = torch.argmax(probs, dim=1)
            
            acc = accuracy_score(list(to_np(output_index)), list(to_np(targets)))
            eval_acc.append(acc)
            eval_loss.append(loss.item())

            
            writer.add_scalar("Evaluation/loss", loss.item(), global_eval_iter)
            writer.add_scalar("Evaluation/acc", acc, global_eval_iter)

            global_eval_iter += 1

    return global_eval_iter, eval_loss, eval_acc


def load_model(args):

    # model = torchvision.models.resnet34()
    model = ResNet18(10)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                momentum=args.momentum,weight_decay=args.decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                                      weight_decay=args.decay)
    else:
        raise NotImplemented("Not implemented optimizer!")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)



    return model, criterion, optimizer, scheduler


def load_cifar10(cifar10_path):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=False, transform=test_transform, download=True)

    return train_data, test_data


args = parsing()
torch.manual_seed(args.seed)
model, criterion, optimizer, scheduler = load_model(args)



cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
train_dataset, test_dataset = load_cifar10(cifar10_path)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
val_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)


if args.model_path is not None:
    save_path = args.save_path
    model_save_path = save_path + 'models/'
else:
    addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
    save_path = f'./run/exp-' + addr + f"_({args.run_index}_{args.noise})_" + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + f'_{args.optimizer}' + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

writer = SummaryWriter(save_path)

train_global_iter = 0
global_eval_iter = 0
best_acc = 0.0
best_loss = np.inf
args.last_lr = args.learning_rate
for epoch in range(0, args.epochs):
    print('epoch', epoch + 1, '/', args.epochs)
    train_global_iter, epoch_loss, epoch_accuracies = train(train_loader, model, train_global_iter, criterion, optimizer, args.device, writer)
    global_eval_iter, eval_loss, eval_acc = test(val_loader, model, global_eval_iter, criterion, args.device, writer)

    writer.add_scalar("Train/avg_loss", np.mean(epoch_loss), epoch)
    writer.add_scalar("Train/avg_acc", np.mean(epoch_accuracies), epoch)
    writer.add_scalar("Evaluation/avg_loss", np.mean(eval_loss), epoch)
    writer.add_scalar("Evaluation/avg_acc", np.mean(eval_acc), epoch)
    writer.add_scalar("Evaluation/avg_acc", np.mean(eval_acc), epoch)
    writer.add_scalar("Train/lr", args.last_lr, epoch)

    print(f"Train/avg_loss: {np.mean(epoch_loss)} Train/avg_acc: {np.mean(epoch_accuracies)} \
          Evaluation/avg_loss: {np.mean(eval_loss)} Evaluation/avg_acc: {np.mean(eval_acc)}")
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path,f'model_params_epoch_{epoch}.pt'))

    if np.mean(eval_loss) < best_loss:
        best_loss = np.mean(eval_loss)
        torch.save(model.state_dict(), os.path.join(save_path,'best_params.pt'))
    
    if np.mean(eval_loss) > best_loss:
        scheduler.step()
        args.last_lr = scheduler.get_last_lr()[0]
    
# os.remove(f"./run_counter/{args.noise}.log")