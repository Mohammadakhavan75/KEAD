import os
import torch
import random
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


from dataset_loader import noise_loader, load_cifar10

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
                        
    parser.add_argument('--lamb', type=float, default=1, help='loss scaling parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')
    

    parser.add_argument('--run_index', default=0, type=int, help='run index')
    parser.add_argument('--one_class_idx', default=None, type=int, help='select one class index')
    parser.add_argument('--auc_cal', action="store_true", help='check if auc calculate')
    parser.add_argument('--tail_positive', default=None, type=float, help='using tail data as negative')
    parser.add_argument('--tail_negative', default=None, type=float, help='using tail data as negative')
    parser.add_argument('--tail_normal', default=None, type=float, help='using tail data as negative')
    parser.add_argument('--temperature', default=0.5, type=float, help='using tail data as negative')
    
    args = parser.parse_args()

    return args


def train_one_class(train_loader, positives, negetives, shuffle_loader,
                     net, train_global_iter, criterion, optimizer, device, writer, stage):

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
    

    a_ = list(negetives)
    b_ = []
    for aa_ in a_:
        b_.append(aa_[0])
    negative_imgs_list = torch.concatenate(b_)

    a_ = list(positives)
    b_ = []
    for aa_ in a_:
        b_.append(aa_[0])
    positives_imgs_list = torch.concatenate(b_)

    a_ = list(shuffle_loader)
    b_ = []
    for aa_ in a_:
        b_.append(aa_[0])
    shuffle_imgs_list = torch.concatenate(b_)

    # for normal, ps, ns, sl in zip(train_loader, positives, negetives, shuffle_loader):
    for normal in zip(train_loader):

        imgs, labels = normal
        # positive_imgs, _ = ps
        # negative_imgs, _ = ns
        # positive_noraml_img, _ = sl

        negative_imgs = negative_imgs_list[torch.randint(0, len(negative_imgs_list), size=(2,))]
        positive_imgs = positives_imgs_list[torch.randint(0, len(positives_imgs_list), size=(1,))]
        positive_noraml_features = shuffle_imgs_list[torch.randint(0, len(shuffle_imgs_list), size=(1,))]

        imgs, labels = imgs.to(args.device), labels.to(args.device)
        positive_imgs, negative_imgs, positive_noraml_img = \
            positive_imgs.to(args.device), negative_imgs.to(args.device), positive_noraml_img.to(args.device)

        optimizer.zero_grad()
        _, normal_features = model(imgs, True)
        _, positive_features = model(positive_imgs, True)
        _, negative_features = model(negative_imgs, True)
        _, positive_noraml_features = model(positive_noraml_img, True)

        positive_features = positive_features[-1]
        normal_features = normal_features[-1]
        negative_features = negative_features[-1]
        positive_noraml_features = positive_noraml_features[-1]
        
        # Calculate loss contrastive for layer
        loss_contrastive = 0
        # for norm_f, pos_f, neg_f, sl_f in zip(normal_features, positive_features, negative_features, positive_noraml_features):
        for norm_f in zip(normal_features):
            # pos_sl_f = torch.stack([pos_f, sl_f])
            # loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, sl_f, temperature=args.temperature))
            pos_sl_f = torch.stack([positive_features[0], positive_noraml_features[0]])
            loss_contrastive = loss_contrastive + torch.sum(contrastive(
                norm_f, pos_sl_f, negative_features, temperature=args.temperature))
            

        # loss_ce = criterion(preds, labels)
        loss = loss_contrastive
        
        epoch_loss['loss'].append(loss.item())
        epoch_loss['contrastive'].append(loss.item())

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss.item(), train_global_iter)

        loss.backward()
        optimizer.step()

    return train_global_iter, epoch_loss, epoch_accuracies 


def test_one_class(eval_loader, positives, negetives, net, global_eval_iter, criterion, device, writer, stage):

    print("Evaluating...")
    net = net.to(device)
    net.eval()  # enter train mode

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
    with torch.no_grad():
        for normal, ps, ns in zip(eval_loader, positives, negetives):
            
            imgs, labels = normal         
            positive_imgs, _ = ps
            negative_imgs, _ = ns
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            positive_imgs, negative_imgs = positive_imgs.to(args.device), negative_imgs.to(args.device)

            preds, normal_features = model(imgs, True)
            positive_preds, positive_features = model(positive_imgs, True)
            negative_preds, negative_features = model(negative_imgs, True)

            # normal_probs = torch.softmax(preds, dim=1)
            # positive_probs = torch.softmax(positive_preds, dim=1)
            # negative_probs = torch.softmax(negative_preds, dim=1)

            # Calculate loss contrastive for layer
            loss_contrastive = 0
            for norm_f, pos_f, neg_f in zip(normal_features[-1], positive_features[-1], negative_features[-1]):
                loss_contrastive = loss_contrastive + torch.sum(contrastive(norm_f, pos_f, neg_f, temperature=args.temperature))

            # loss_ce = criterion(preds, labels)

            if stage == 'feature_extraction':
                loss = loss_contrastive
            if stage == 'classifier':
                loss = loss_ce

            # normal_output_index = torch.argmax(normal_probs, dim=1)
            # positive_output_index = torch.argmax(positive_probs, dim=1)
            # negative_output_index = torch.argmax(negative_probs, dim=1)
            # acc_normal = accuracy_score(list(to_np(normal_output_index)), list(to_np(labels)))
            # acc_positive = accuracy_score(list(to_np(positive_output_index)), list(to_np(labels)))
            # acc_negative = accuracy_score(list(to_np(negative_output_index)), list(to_np(labels)))

            # Logging section
            # epoch_accuracies['normal'].append(acc_normal)
            # epoch_accuracies['positive'].append(acc_positive)
            # epoch_accuracies['negative'].append(acc_negative)
            epoch_loss['loss'].append(loss.item())
            # epoch_loss['ce'].append(loss_ce.item())
            epoch_loss['contrastive'].append(loss_contrastive.item())

            global_eval_iter += 1
            writer.add_scalar("Evaluation/loss", loss.item(), global_eval_iter)
            # writer.add_scalar("Evaluation/loss_ce", loss_ce.item(), global_eval_iter)
            writer.add_scalar("Evaluation/loss_contrastive", loss_contrastive.item(), global_eval_iter)
            # writer.add_scalar("Evaluation/acc_normal", acc_normal, global_eval_iter)
            # writer.add_scalar("Evaluation/acc_positive", acc_positive, global_eval_iter)
            # writer.add_scalar("Evaluation/acc_negative", acc_negative, global_eval_iter)


    return global_eval_iter, epoch_loss, epoch_accuracies


def eval_auc_one_class(eval_in, eval_out, net, global_eval_iter, criterion, device):

    print("Evaluating...")
    net = net.to(device)
    net.eval()  # enter train mode

    # track train classification accuracy
    epoch_accuracies = {
        'acc': [],
        'auc': [],
    }
    epoch_loss = {
        'loss': [],
    }
    with torch.no_grad():
        for data_in, data_out in zip(eval_in, eval_out):
            
            inputs_in, targets_in = data_in
            targets_in_auc = [1 for _ in targets_in]
            targets_in_auc = torch.tensor(targets_in_auc, dtype=torch.float32)

            inputs_out, targets_out = data_out
            targets_out_auc = [0 for _ in targets_out]
            targets_out_auc = torch.tensor(targets_out_auc, dtype=torch.float32)

            
            inputs_ = torch.cat((inputs_in, inputs_out), dim=0)
            targets_auc_ = torch.cat((targets_in_auc, targets_out_auc), dim=0)
            targets_ = torch.cat((targets_in, targets_out), dim=0)
            random_indices = torch.randperm(inputs_.size(0))
            inputs = inputs_[random_indices]
            targets = targets_[random_indices]
            targets_auc = targets_auc_[random_indices]

            inputs , targets = inputs.to(device) , targets.to(device)

            preds, normal_features = net(inputs, True)            

            loss = criterion(preds, targets)
            
            acc = accuracy_score(list(to_np(torch.argmax(torch.softmax(preds, dim=1), axis=1))), list(to_np(targets)))
            auc = roc_auc_score(to_np(targets_auc), to_np(torch.argmax(torch.softmax(preds, dim=1), axis=1) == targets) )

            # Logging section
            epoch_accuracies['acc'].append(acc)
            epoch_accuracies['auc'].append(auc)
            epoch_loss['loss'].append(loss.item())

    return epoch_loss, epoch_accuracies


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

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    
    milestones = [3, 7, 13, 30, 70]  # Epochs at which to decrease learning rate
    gamma = 0.1  # Factor by which to decrease learning rate at milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)



    return model, criterion, optimizer, scheduler


def evaluation(model, save_path, cifar10_path, args):
    from dataset_loader import get_subclass_dataset
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    test_data = torchvision.datasets.CIFAR10(cifar10_path, train=False, transform=test_transform, download=True)

    idxes = [i for i in range(10)]
    idxes.pop(args.one_class_idx)

    eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)
    eval_out_data = get_subclass_dataset(test_data, idxes)

    eval_in = DataLoader(eval_in_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    model_path = save_path + 'best_params.pt'
    model.load_state_dict(torch.load(model_path))

    global_eval_iter, eval_loss, eval_acc = eval_auc_one_class(eval_in, eval_out, model, global_eval_iter, criterion, args.device)

    print(f"Evaluation_Results/avg_loss: {np.mean(eval_loss['loss'])}, Evaluation_Results/avg_acc: {np.mean(eval_acc['acc'])}, Results/avg_auc: {np.mean(eval_acc['auc'])}")


args = parsing()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model, criterion, optimizer, scheduler = load_model(args)

cifar10_path = '/storage/users/makhavan/CSI/finals/datasets/data/'
train_loader, test_loader, shuffle_loader = load_cifar10(cifar10_path, one_class_idx=args.one_class_idx, tail_normal=args.tail_normal)

print("Start Loading noises")
train_positives, train_negetives, test_positives, test_negetives = \
    noise_loader(one_class_idx=args.one_class_idx, tail_negative=args.tail_negative, tail_positive=args.tail_positive)
print("Loading noises finished!")

if args.model_path is not None:
    save_path = args.save_path
    model_save_path = save_path + 'models/'
else:
    addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
    save_path = f'./run/exp-' + addr + f'_{args.learning_rate}' + f'_{args.lr_update_rate}' + f'_{args.lr_gamma}' + \
    f'_{args.optimizer}' + f'_epochs_{args.epochs}' + f'_one_class_idx_{args.one_class_idx}' + \
        f'_temprature_{args.temperature}' + f'_tailpos_{args.tail_positive}' + f'_tailneg_{args.tail_negative}' + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

writer = SummaryWriter(save_path)

train_global_iter = 0
global_eval_iter = 0
best_acc = 0.0
best_loss = np.inf
args.last_lr = args.learning_rate

stage = 'feature_extraction'

for epoch in range(0, args.epochs):
    # if epoch + 3 > args.epochs:
    #     stage = 'classifier'
    print('epoch', epoch + 1, '/', args.epochs)
    if args.one_class_idx != None:
        if stage == 'feature_extraction':
            model.linear.requires_grad_ = False
            train_global_iter, epoch_loss, epoch_accuracies = train_one_class(train_loader, train_positives, train_negetives, shuffle_loader, model, train_global_iter, criterion, optimizer, args.device, writer, stage)
            global_eval_iter, eval_loss, eval_acc = test_one_class(test_loader, test_positives, test_negetives, model, global_eval_iter, criterion, args.device, writer, stage)
        if stage == 'classifier':
            print("\nSTAGE changed from feature_extraction into classifier\n\n")
            model.linear.requires_grad_ = True
            model.layer1.requires_grad_ = False
            model.layer2.requires_grad_ = False
            model.layer3.requires_grad_ = False
            model.layer4.requires_grad_ = False
            model.conv1.requires_grad_ = False
            train_global_iter, epoch_loss, epoch_accuracies = train_one_class(train_loader, train_positives, train_negetives, model, train_global_iter, criterion, optimizer, args.device, writer, stage)
            global_eval_iter, eval_loss, eval_acc = test_one_class(test_loader, test_positives, test_negetives, model, global_eval_iter, criterion, args.device, writer, stage)

    writer.add_scalar("AVG_Train/avg_loss", np.mean(epoch_loss['loss']), epoch)
    writer.add_scalar("AVG_Train/avg_loss_contrastive", np.mean(epoch_loss['contrastive']), epoch)
    writer.add_scalar("AVG_Evaluation/avg_loss", np.mean(eval_loss['loss']), epoch)
    writer.add_scalar("AVG_Evaluation/avg_loss_contrastive", np.mean(eval_loss['contrastive']), epoch)

    writer.add_scalar("Train/lr", args.last_lr, epoch)

    print(f"Train/avg_loss: {np.mean(epoch_loss['loss'])} Evaluation/avg_loss: {np.mean(eval_loss['loss'])}")
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path,f'model_params_epoch_{epoch}.pt'))

    if np.mean(eval_loss['loss']) < best_loss:
        best_loss = np.mean(eval_loss['loss'])
        torch.save(model.state_dict(), os.path.join(save_path,'best_params.pt'))
    
    # if np.mean(eval_loss['loss']) > best_loss:
    #     scheduler.step()
        # if args.last_lr != scheduler.get_last_lr()[0]:
        #     scheduler.step_size = scheduler.step_size + 3
        #     print(f"scheduler step_size has increased into: {scheduler.step_size}")

        args.last_lr = scheduler.get_last_lr()[0]

    if args.auc_cal and epoch % 11 == 1:
        evaluation(model, save_path, cifar10_path, args)

    scheduler.step()

torch.save(model.state_dict(), os.path.join(save_path,'last_params.pt'))
if args.auc_cal:
    evaluation(model, save_path, cifar10_path, args)
    exit()