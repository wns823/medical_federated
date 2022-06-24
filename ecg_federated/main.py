import torch
import numpy as np
import wandb, random, argparse
from ecg_federated import *

def set_seed(seed) :
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    tp = lambda x:list(map(str, x.split('.')))
    parser.add_argument('--data_list', type=tp, default="ChapmanShaoxing.CPSC2018.Ga.Ningbo.PTBXL", help="data list")
    
    parser.add_argument('--epochs', type=int, default=200, help="Epochs")
    parser.add_argument('--seed', type=int, default=1234, help="seed")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="workers of dataloader")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay")
    parser.add_argument('--scheduler_step', type=int, default=1, help="scheduler step")
    parser.add_argument('--test', action='store_true', help ='test the model')
    parser.add_argument('--max_norm', type=float, default=5.0, help="max norm")

    parser.add_argument('--load_dir', type=str, default='federated_ecg_manifest', help="load dir")
    parser.add_argument('--save_path', type=str, default='ecg_checkpoint', help="save path")

    parser.add_argument('--model_type', type=str, default="resnet", help="model type")

    parser.add_argument('--communications', type=int, default=200, help="communications")
    parser.add_argument('--local_epochs', type=int, default=1, help="local epochs")
    parser.add_argument('--algorithm', type=str, default='fedavg', help="fedavg|fedprox|fedbn|fedadam|fedadagrad|fedyogi") 
    parser.add_argument('--mu', type=float, default=0.01, help="fedprox hyperparameter") # 0.5

    parser.add_argument('--server_learning_rate', type=float, default=0.01, help="fedopt server learning rate") # 1e-1
    parser.add_argument('--tau', type=float, default=0.01, help="fedopt tau") # 0.0

    parser.add_argument('--beta_1', type=float, default=0.9, help="fedopt beta 1") # 0.5
    parser.add_argument('--beta_2', type=float, default=0.99, help="fedprox hyperparameter") # 0.5

    # 0.001 0.01 0.01
    parser.add_argument('--feddyn_alpha', type=float, default=0.1, help="feddyn alpha") # 0.5

    args = parser.parse_args()
    set_seed(args.seed)

    if args.test :
        wandb_run = None
    else :
        wandb_run = wandb.init(project="ECG federated")
        wandb_run.name = f"{args.algorithm}_{args.communications}_{args.local_epochs}_{args.model_type}_{args.seed}"

    federated_main(args, wandb_run=wandb_run)