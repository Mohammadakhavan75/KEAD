import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--save_path', type=str,
                        default=None, help='Path to save files.')
    parser.add_argument('--device', type=str, 
                        default="cuda", help='cuda or cpu.')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Select gpu number')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='Batch size.')
    
    # Dataset Configuration
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='cifar10-cifar100-svhn')
    parser.add_argument('--one_class_idx', default=None, type=int,
                        help='select one class index')
    

    # Optimizer Configuration
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='The initial learning rate.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                        help='The initial learning rate.')
    parser.add_argument('--last_lr', type=float, default=0,
                        help='The gamma param for updating learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-6,
                        help='Weight decay (L2 penalty).')

    # Model Configuration
    parser.add_argument('--projection_head', default=0, type=int,
                        help='Weight of Contrastive loss')
    parser.add_argument('--fc_available', default=0, type=int,
                        help='Weight of Contrastive loss')
    parser.add_argument('--img_size', default=32, type=int,
                        help='image size selection')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='resnet model selection')

    parser.add_argument('--temperature', '--tau', default=0.5, type=float,
                        help='chaning temperature of contrastive loss')
    parser.add_argument('--k_view', '-K', type=int, default=4,
                        help='Number of stochastic views')
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--exp_idx', default=0, type=int,
                        help='run index')

    args = parser.parse_args()

    return args
