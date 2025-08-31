import os
from models.resnet import ResNet18, ResNet50
import torch
from natsort import natsorted

def load_model(args):
    if args.model == 'resnet18':
        model = ResNet18(img_size=args.img_size, num_classes=1, classification_head=args.fc_available, proj_head=args.proj_head, proj_dim=args.proj_dim) # num_class is 1 for binary anomaly classification
    elif args.model == 'resnet50':
        model = ResNet50(img_size=args.img_size, num_classes=1, classification_head=args.fc_available, proj_head=args.proj_head, proj_dim=args.proj_dim)
    else:
        raise NotImplementedError("Not implemented model!")
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                momentum=args.momentum,weight_decay=args.decay)
        lr_decay_gamma = 0.1
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                                    weight_decay=args.decay)
        lr_decay_gamma = 0.1
    elif args.optimizer == 'lars':
        from torchlars import LARS
        base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
        lr_decay_gamma = 0.1
    else:
        raise NotImplementedError("Not implemented optimizer!")

    # if args.resume:
    #     model_folder = args.save_path + 'models/'
    #     model_name = natsorted(os.listdir(model_folder))[-1]
    #     args.from_epoch = int(model_name.split('_')[-1].split('.')[0])
    #     model.load_state_dict(torch.load(os.path.join(model_folder, model_name), weights_only=True))

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * args.epochs), int(0.75 * args.epochs)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError("Not implemented learning rate scheduler!")

    return model, optimizer, scheduler
