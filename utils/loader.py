import os
from models.resnet import ResNet18, ResNet50
import torch
from natsort import natsorted

def load_model(args):
    if args.model == 'resnet18':
        model = ResNet18(img_size=args.img_size, num_classes=1, fc_available=args.fc_available, proj_head=args.proj_head, proj_dim=args.proj_dim) # num_class is 1 for binary anomaly classification
    elif args.model == 'resnet50':
        model = ResNet50(img_size=args.img_size, num_classes=1, fc_available=args.fc_available, proj_head=args.proj_head, proj_dim=args.proj_dim)
    else:
        raise NotImplementedError("Not implemented model!")
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                momentum=args.momentum,weight_decay=args.decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                                    weight_decay=args.decay)
    else:
        raise NotImplementedError("Not implemented optimizer!")

    # if args.resume:
    #     model_folder = args.save_path + 'models/'
    #     model_name = natsorted(os.listdir(model_folder))[-1]
    #     args.from_epoch = int(model_name.split('_')[-1].split('.')[0])
    #     model.load_state_dict(torch.load(os.path.join(model_folder, model_name), weights_only=True))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    return model, optimizer, scheduler
