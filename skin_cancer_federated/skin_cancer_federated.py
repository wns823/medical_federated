import os, wandb
import torch
import torch.nn as nn
from torch import optim
from models import *
import log
from skin_cancer_datasets import get_federated_dataset
from copy import deepcopy
from communication_func import *
from skin_cancer_core import *

def update_best_models(args, best_models, local_models) :
    for i in range(len(local_models)) :
        best_models[i] = deepcopy(local_models[i])
    return best_models

def save_models(args, server_model, local_models, save_path, best=False):
    model_list = {}
    
    if best == True :
        model_list = {'server_model' : server_model.state_dict()}

        for i in range(len(local_models)) :
            model_list[f'model_{i}'] = local_models[i].state_dict()

        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_best_{args.algorithm}_{args.model_type}_{args.server_learning_rate}_{args.learning_rate}_{args.tau}_{args.seed}.pt" ) )    
        else :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_best_{args.algorithm}_{args.model_type}_{args.seed}.pt" ) )
        
    else :
        model_list = {'server_model' : server_model.state_dict()}

        for i in range(len(local_models)) :
            model_list[f'model_{i}'] = local_models[i].state_dict()

        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_{args.algorithm}_{args.model_type}_{args.server_learning_rate}_{args.learning_rate}_{args.tau}_{args.seed}.pt" ) )    
        else :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_{args.algorithm}_{args.model_type}_{args.seed}.pt" ) )

def initialization(args, model):
    factor = deepcopy(model)

    for key in factor.state_dict().keys():
        if 'num_batches_tracked' in key:
            pass;
        else:
            temp = torch.zeros_like(factor.state_dict()[key])
            # temp.new_full(temp.shape, args.tau ** 2) ## 
            factor.state_dict()[key].data.copy_(temp)

    return factor

###############################################################################################################

def initialization_prev_grad(args, model):
    
    factor = deepcopy(model)

    for key in factor.state_dict().keys():
        if 'num_batches_tracked' in key:
            pass;
        else:
            temp = torch.zeros_like(factor.state_dict()[key])
            factor.state_dict()[key].data.copy_(temp)

    prev_grads = [deepcopy(factor) for idx in range(len(args.clients))]

    return prev_grads

def get_param_tensor(target):
    temp_tensor = None
    for name, param in target.named_parameters():
        if not isinstance(temp_tensor, torch.Tensor):
            temp_tensor = param.view(-1).clone()
        else:
            temp_tensor = torch.cat((temp_tensor, param.view(-1).clone()), dim=0)                    
    return temp_tensor

def get_grad_diff( prev_grads, current_model, cld_model):
    with torch.no_grad():
        # aggregate params
        for key in cld_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                pass
            else:
                temp = current_model.state_dict()[key] - cld_model.state_dict()[key]
                temp += prev_grads.state_dict()[key]
                prev_grads.state_dict()[key].data.copy_(temp)
                
    return prev_grads

###############################################################################################################


def skin_cancer_main(args, wandb_run=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_name = "_".join(args.clients)
    SAVE_PATH = f'{args.save_path}/federated/{args.algorithm}_{args.communication}_{args.local_epochs}_{save_name}'

    if not os.path.exists( SAVE_PATH ):
        os.makedirs( SAVE_PATH )

    os.makedirs(os.path.join(SAVE_PATH, "logs"), exist_ok=True)
    log_file = os.path.join(SAVE_PATH, "logs", "test_{}_{}_{}_{}_{}.txt".format( args.algorithm , args.communication , args.local_epochs, save_name, args.seed))


    train_loaders, valid_loaders, test_loaders, client_weights = get_federated_dataset(args)


    server_model = SkinModel(args.model_type).to(device)

    param_name = []
    for name, param in server_model.named_parameters() :
        param_name.append(name)

    optimizer = optim.Adam(server_model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss() 
    # criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(client_weights))

    local_models = [deepcopy(server_model).to(device) for idx in range(len(args.clients))]
    best_scores = 0.0
    best_models = [ deepcopy(server_model) for idx in range(len(args.clients)) ]
    patience = 0
    best_communication_rounds = 0

    if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
        m_t = initialization(args, server_model)
        v_t = initialization(args, server_model)

    if args.algorithm == "feddyn" :
        prev_grads = initialization_prev_grad(args, server_model)
        cld_model = deepcopy(server_model)


    if args.test :

        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            save_model_path = os.path.join( f"{SAVE_PATH}", f"checkpoint_best_{args.algorithm}_{args.model_type}_{args.server_learning_rate}_{args.learning_rate}_{args.tau}_{args.seed}.pt")
        else :
            save_model_path = os.path.join( f"{SAVE_PATH}", f"checkpoint_best_{args.algorithm}_{args.model_type}_{args.seed}.pt")

        checkpoint = torch.load(f'{save_model_path}')
        for i in range(len(args.clients)) :
            local_models[i].load_state_dict(checkpoint[f'model_{i}'])

        server_model.load_state_dict(checkpoint['server_model'])

        for c in range(len(args.clients)) :
            print()

            print(f"########## {c} client's test result ##########")

            if args.algorithm != "feddyn" :
                test_loss, test_pr_auc, test_roc_auc = test_single(local_models[c], test_loaders[c], criterion, device=device)
            else :
                test_loss, test_pr_auc, test_roc_auc = test_single(server_model, test_loaders[c], criterion, device=device)

            print("TEST loss : ", test_loss)
            print("TEST PR AUC : ", test_pr_auc)
            print("TEST ROC AUC : ", test_roc_auc)
            print()

        print()
        exit()


    with open(log_file, "w") as file:
        file.write("")

    ############################################################################################################################################################
    for comms in range(args.communication):
    
        if args.algorithm != "feddyn" :
            optimizers = [ torch.optim.Adam( local_models[idx].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) for idx in range(len(args.clients)) ]
        else :
            optimizers = [ torch.optim.Adam( local_models[idx].parameters(), lr=args.learning_rate, weight_decay=args.feddyn_alpha + args.weight_decay) for idx in range(len(args.clients)) ]

        for c in range(len(args.clients)) :
            model, train_loader, optimizer = local_models[c], train_loaders[c], optimizers[c]
            
            if args.algorithm == "feddyn" :
                prev_grad_tensor = get_param_tensor(prev_grads[c])
                server_model_tensor = get_param_tensor(server_model)

            for epoch in range(args.local_epochs) :
                total_steps = epoch + comms * args.local_epochs

                if (args.algorithm == "fedprox" or args.algorithm == "fedpxn") and comms > 0 :
                    train_loss = train_fedprox(args, model, server_model, train_loader, optimizer, criterion, param_name, device=device)
                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )
                elif args.algorithm == "feddyn" :
                    train_loss = train_feddyn(args, model, train_loader, optimizer, criterion, server_model_tensor, prev_grad_tensor, device=device)
                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )
                else :
                    train_loss = train_single(model, train_loader, optimizer, criterion, device=device)

                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )

            if args.algorithm == "feddyn" :
                prev_grads[c] = get_grad_diff( prev_grads[c], model, cld_model)

        ###################################################################################################
        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            server_model, local_models, m_t, v_t = communication_fedopt(args, server_model, local_models, client_weights, m_t, v_t)
        elif args.algorithm == "feddyn" :
            server_model, local_models, cld_model = communication_FedDyn(args, server_model, local_models, client_weights, cld_model, prev_grads)        
        else :
            server_model, local_models = communication(args, server_model, local_models, client_weights)
        ###################################################################################################

        total_steps = (comms + 1) * args.local_epochs
        step_scores = 0.0

        for c in range(len(args.clients)) :

            if args.algorithm != "feddyn" :
                model, valid_loader = local_models[c], valid_loaders[c]
            else :
                model, valid_loader = server_model, valid_loaders[c]

            valid_loss, valid_pr_auc, valid_roc_auc = test_single(model, valid_loader, criterion, device=device)
            step_scores += valid_roc_auc

            print(f"{c} epoch's valid loss : ", valid_loss)
            print(f"{c} epoch's valid PR AUC : ", valid_pr_auc)
            print(f"{c} epoch's valid ROC AUC : ", valid_roc_auc)

            if wandb_run :
                wandb_run.log( {f'{c}_valid_loss ': valid_loss, 'total_step' : total_steps} )
                wandb_run.log( {f'{c}_valid_pr_auc ': valid_pr_auc, 'total_step' : total_steps} )
                wandb_run.log( {f'{c}_valid_roc_auc ': valid_roc_auc, 'total_step' : total_steps} )

        ###########################################################################################################

        # save best model
        step_scores = step_scores / len(args.clients)
        if step_scores > best_scores :
            patience = 0
            best_scores = step_scores
            best_models = update_best_models(args, best_models, local_models)
            save_models(args, server_model, best_models, SAVE_PATH, best=True)
            best_communication_rounds = comms + 1
        else :
            patience += 1

        save_models(args, server_model, local_models, SAVE_PATH, best=False)
    
    print("patience : ", patience)
    print("communication round of best model : ", best_communication_rounds)
    print("relative global epochs of best model : ", best_communication_rounds * args.local_epochs)

    if wandb_run :
        wandb_run.log( {'patience': patience} )
        wandb_run.log( {'communication_round_of_best_model': best_communication_rounds} )
        wandb_run.log( {'relative_global_epochs_of_best_model': best_communication_rounds * args.local_epochs} )
