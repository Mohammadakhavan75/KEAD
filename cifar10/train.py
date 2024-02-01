import os
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
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


def train(loader, net, train_global_iter, criterion, optimizer, device, writer):

    print("Traning...")
    net = net.to(device)
    net.train()  # enter train mode

    # track train classification accuracy
    epoch_accuracies = []
    epoch_loss = []
    for inputs, targets in tqdm(loader):
        
        inputs , targets = inputs.to(device) , targets.to(device)

        optimizer.zero_grad()
        preds = net(inputs)
        probs = torch.softmax(preds, dim=1)

        loss = criterion(preds, targets)
        
        output_index = torch.argmax(probs, dim=1)
        acc = accuracy_score(list(to_np(output_index)), list(to_np(targets)))
        epoch_accuracies.append(acc)
        epoch_loss.append(loss.item())

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss.item(), train_global_iter)
        writer.add_scalar("Train/acc", acc, train_global_iter)

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

    model = torchvision.models.resnet34()
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