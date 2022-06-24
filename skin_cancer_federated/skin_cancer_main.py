import torch
import argparse, os
import numpy as np
import random
import wandb
from skin_cancer_federated import skin_cancer_main

def add_fit_args(parser):
    
    tp = lambda x:list(map(str, x.split('.')))
    parser.add_argument('--clients', type=tp, default="barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt", help="client list")

    parser.add_argument('--model_type', type=str, default='efficientnet', help="efficientnet|efficientnet_gn")    

    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout")

    parser.add_argument('--save_path', type=str, default='checkpoint', help="Checkpoint directory")

    parser.add_argument('--data_path', type=str, default='/home/data_storage', help="preprocessed data path")

    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--learning_rate_step', type=float, default=1, help="Learning rate step") # 10
    parser.add_argument('--learning_rate_decay', type=float, default=0.15, help="Learning rate step") # 0.5

    parser.add_argument('--test', action='store_true', help ='test the pretrained model')

    parser.add_argument('--communication', type = int, default=300, help = 'iterations for communication') # 100
    parser.add_argument('--local_epochs', type = int, default=1, help = 'optimization iters in local worker between communication') ## local epoch
    parser.add_argument('--algorithm', type = str, default='fedavg', help='fedavg | fedprox | fedbn | fedadam | fedadagrad | fedyogi')

    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')

    parser.add_argument('--seed', type = int, default=0, help = 'seed')
    parser.add_argument('--workers', type = int, default=8, help = 'workers')

    parser.add_argument('--server_learning_rate', type=float, default=0.01, help="fedopt server learning rate") # 1e-1
    parser.add_argument('--tau', type=float, default=0.01, help="fedopt tau") # 0.0

    parser.add_argument('--beta_1', type=float, default=0.9, help="fedopt beta 1") # 0.5
    parser.add_argument('--beta_2', type=float, default=0.99, help="fedprox hyperparameter") # 0.5

    parser.add_argument('--feddyn_alpha', type=float, default=0.01, help="feddyn alpha") # 0.5


    args = parser.parse_args()
    return args

def set_seed(seed) :
    np.random.seed(1234)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True # False


if __name__ == "__main__" :    
    args = add_fit_args(argparse.ArgumentParser())

    set_seed(args.seed)

    if args.test :
        wandb_run = None
    else : ######################################
        naming = "_".join(args.clients)
        wandb_run = wandb.init(project=f"skin image federated")
        wandb_run.name = f"{args.algorithm}_{args.communication}_{args.local_epochs}_{args.model_type}_{naming}_{args.seed}"

    skin_cancer_main(args, wandb_run=wandb_run)
    