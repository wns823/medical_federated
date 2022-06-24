import torch
import numpy as np
import wandb, random, argparse
from ehr_federated import *

def set_seed(seed) :
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234, help="seed")

    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--learning_rate_step', type=float, default=1, help="Learning rate step") # 10
    parser.add_argument('--learning_rate_decay', type=float, default=0.15, help="Learning rate step") # 0.5
    parser.add_argument('--epochs', type=int, default=50, help="Epochs")

    parser.add_argument('--model_type', type=str, default='transformer_ln', help="transformer_ln|transformer_gn")    
    parser.add_argument('--task', type=str, default='disch_48h', help="disch_24h|disch_48h|mort_24h|mort_48h|Final Acuity Outcome|LOS")

    parser.add_argument('--data_path', type=str, default='data_storage', help="data path")    
    parser.add_argument('--save_path', type=str, default='eicu_checkpoint', help="save path")    
    parser.add_argument('--test', action='store_true', help ='test the model')

    tp = lambda x:list(map(str, x.split('.')))
    parser.add_argument('--hospital_id', type=tp, default="73.264.420.243.458", help="hospital id list")

    parser.add_argument('--communications', type=int, default=500, help="communications")
    parser.add_argument('--local_epochs', type=int, default=1, help="local epochs")
    parser.add_argument('--algorithm', type=str, default='fedavg', help="fedavg|fedprox|fedbn|fedadam|fedadagrad|fedyogi") 
    parser.add_argument('--mu', type=float, default=0.01, help="fedprox hyperparameter") # 0.5

    parser.add_argument('--server_learning_rate', type=float, default=0.01, help="fedopt server learning rate") # 1e-1
    parser.add_argument('--tau', type=float, default=0.01, help="fedopt tau") # 0.0

    parser.add_argument('--beta_1', type=float, default=0.9, help="fedopt beta 1") # 0.5
    parser.add_argument('--beta_2', type=float, default=0.99, help="fedprox hyperparameter") # 0.5

    parser.add_argument('--feddyn_alpha', type=float, default=0.1, help="feddyn alpha") # 0.5

    args = parser.parse_args()
    set_seed(args.seed)

    if args.test :
        wandb_run = None
    else :
        wandb_run = wandb.init(project="EHR federated" , entity='kaist-edlab-nmt')
        naming = "max_" + str(len(args.hospital_id))
        wandb_run.name = f"{args.algorithm}_{args.communications}_{args.local_epochs}_{args.model_type}_{naming}_{args.task}_{args.seed}"

    ehr_federated_main(args, wandb_run=wandb_run)
