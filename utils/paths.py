import os

def create_path(args):
    root_addr = f'./run/exp_{args.exp_idx}/'
    if args.exp_idx == 0:
        while os.path.exists(root_addr):
            args.exp_idx += 1
            root_addr = root_addr.split('_')[0] + '_' + str(args.exp_idx) + '/'
        
        os.makedirs(root_addr)


    save_path = root_addr + f'{args.dataset}' + f'_one_class_idx_{args.one_class_idx}' + \
                f'_{args.model}' + f'_{args.img_size}' + f'_{args.optimizer}' + f'_lr_{args.learning_rate}' + \
                f'_epochs_{args.epochs}' + f'_tau_{args.temperature}' + f'_seed_{args.seed}' + f'_k_view_{args.k_view}' + '/'
    
    model_save_path = save_path + 'models/'
    os.makedirs(model_save_path, exist_ok=True)

    return model_save_path, save_path
