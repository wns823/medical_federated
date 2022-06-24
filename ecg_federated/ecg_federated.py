import os, math
import torch
import torch.nn as nn
from fairseq_signals.data import FileECGDataset
from fairseq_signals.data.ecg import ecg_utils
import numpy as np
import torch.nn.functional as F
from core import *
from base_model import load_crnn
from torch_ecg.models.loss import AsymmetricLoss
import wandb, random, argparse
from copy import deepcopy
from communication_func import *
from torch.utils.data import RandomSampler

def get_split_loader(args, path, split=True) :
    split_dataset = FileECGDataset(
            manifest_path=path,
            sample_rate=500,
            max_sample_size=None,
            min_sample_size=None,
            pad=True,
            pad_leads=False,
            leads_to_load=None,
            label=True,
            normalize=False,
            num_buckets=0,
            compute_mask_indices=False,
            leads_bucket=None,
            bucket_selection="uniform",
            training=split, # True, False
            **{}
        )

    if split == True :
        sampler = RandomSampler(split_dataset)
    else :
        sampler = None

    data_size = len(split_dataset)

    data_loader = torch.utils.data.DataLoader(
            split_dataset,
            batch_size=args.batch_size,
            collate_fn = split_dataset.collator,
            sampler = sampler,
            num_workers = args.num_workers,
        )
    
    return data_loader, data_size


def get_dataloader(args) :
    
    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    for dataname in args.data_list :
        common_dir = f"{args.load_dir}/{dataname}/cinc"
        train_loader, train_size = get_split_loader( args, os.path.join( common_dir, "train.tsv" ) , split=True)
        valid_loader, valid_size = get_split_loader( args, os.path.join( common_dir, "valid.tsv" ) , split=False)
        test_loader, test_size = get_split_loader( args, os.path.join( common_dir, "test.tsv" ) , split=False)

        train_loaders.append( train_loader )
        valid_loaders.append( valid_loader )
        test_loaders.append( test_loader )
        client_weights.append( train_size )

    client_weighted = [ c_weight / sum(client_weights) for c_weight in client_weights ]

    return train_loaders, valid_loaders, test_loaders, client_weighted


def save_models(args, models, save_path, best=False) :
    model_list = {}
    for idx in range(len(args.data_list)) :
        model_name = 'model_' + str(idx)
        model_list[model_name] = models[idx].state_dict()

    if best == True :
        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_best_{args.model_type}_{args.learning_rate}_{args.server_learning_rate}_{args.tau}_{args.seed}.pt" ) )    
        else :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_best_{args.model_type}_{args.seed}.pt" ) )
    else :
        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_{args.model_type}_{args.learning_rate}_{args.server_learning_rate}_{args.tau}_{args.seed}.pt" ) )    
        else :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_{args.model_type}_{args.seed}.pt" ) )


def update_best_models(args, best_models, models) :
    for idx in range(len(args.data_list)) :
        best_models[idx] = deepcopy(models[idx])
    return best_models


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

def initialization_prev_grad(args, model):
    
    factor = deepcopy(model)

    for key in factor.state_dict().keys():
        if 'num_batches_tracked' in key:
            pass;
        else:
            temp = torch.zeros_like(factor.state_dict()[key])
            factor.state_dict()[key].data.copy_(temp)

    prev_grads = [deepcopy(factor) for idx in range(len(args.data_list))]

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

##############################################################################################################################

def federated_main(args, wandb_run=None) :

    naming = "_".join(args.data_list)

    SAVE_PATH = f'{args.save_path}/federated/{args.algorithm}_{args.communications}_{args.local_epochs}'
    if not os.path.exists( SAVE_PATH ):
        os.makedirs( SAVE_PATH )

    patience = 0
    best_communication_rounds = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes, score_weights = ( ecg_utils.get_physionet_weights("weights.csv") )
    sinus_rhythm_index = ecg_utils.get_sinus_rhythm_index(classes)

    train_loaders, valid_loaders, test_loaders, client_weights = get_dataloader(args)
    
    server_model = load_crnn(args)

    if args.algorithm == "fedpxn" :
        param_name = []
        for name, param in server_model.named_parameters() :
            param_name.append(name)

    server_model = server_model.to(device)

    local_models = [deepcopy(server_model) for idx in range(len(args.data_list))]

    if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
        m_t = initialization(args, server_model)
        v_t = initialization(args, server_model)

    if args.algorithm == "feddyn" :
        prev_grads = initialization_prev_grad(args, server_model)
        cld_model = deepcopy(server_model)

    best_scores = 0.0
    best_models = []
    for idx in range(len(args.data_list)):
        best_models.append( deepcopy(server_model) )

    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")

    if args.test :

        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            checkpoint = torch.load( os.path.join(f'{SAVE_PATH}', f'checkpoint_best_{args.model_type}_{args.learning_rate}_{args.server_learning_rate}_{args.tau}_{args.seed}.pt'))
        else :
            checkpoint = torch.load( os.path.join(f'{SAVE_PATH}', f'checkpoint_best_{args.model_type}_{args.seed}.pt'))

        for client_idx in range(len(args.data_list)):
            local_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])

        print(f"communications : {args.communications}, local epochs : {args.local_epochs}, {args.algorithm} results")

        for c in range(len(args.data_list)) :
            model, test_loader = local_models[c], test_loaders[c]
            print(f"########## {c} client's test result ##########")
            test_loss, test_cinc = naive_test(args, model, test_loader, score_weights, sinus_rhythm_index, criterion, device=device)

            print(f"{c} client test loss : ", test_loss)
            print(f"{c} client test cinc : ", test_cinc)
            print()

        exit()


    for comms in range(args.communications):
    
        if args.algorithm != "feddyn" :
            optimizers = [ torch.optim.AdamW( params=local_models[idx].parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-08, amsgrad=True) for idx in range(len(args.data_list)) ]
        else :
            optimizers = [ torch.optim.AdamW( params=local_models[idx].parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.feddyn_alpha+args.weight_decay, eps=1e-08, amsgrad=True) for idx in range(len(args.data_list)) ]

        schedulers = [ torch.optim.lr_scheduler.OneCycleLR(optimizers[idx], max_lr=2e-3, epochs=args.scheduler_step, steps_per_epoch=len(train_loaders[idx]),) for idx in range(len(args.data_list)) ]

        for c in range(len(args.data_list)) :

            model, train_loader, optimizer, scheduler = local_models[c], train_loaders[c], optimizers[c], schedulers[c]

            if args.algorithm == "feddyn" :
                prev_grad_tensor = get_param_tensor(prev_grads[c])
                server_model_tensor = get_param_tensor(server_model)

            for epoch in range(args.local_epochs) :
                total_steps = epoch + comms * args.local_epochs

                if ( args.algorithm == "fedprox" or args.algorithm == "fedpxn" ) and comms > 0 :
                    if args.algorithm == "fedpxn" :
                        train_loss = train_fedprox(args, server_model, model, train_loader, criterion, optimizer, param_name=param_name, device=device, scheduler=scheduler)
                    else :
                        train_loss = train_fedprox(args, server_model, model, train_loader, criterion, optimizer, param_name=None, device=device, scheduler=scheduler)

                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )
                elif args.algorithm == "feddyn" :
                    train_loss = train_feddyn(args, model, server_model_tensor, train_loader, criterion, optimizer, prev_grad_tensor, device=device, scheduler=scheduler)
                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )
                else :
                    train_loss = naive_train(args, model, train_loader, optimizer, criterion, device=device, scheduler=scheduler)
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

        for c in range(len(args.data_list)) :
            if args.algorithm != "feddyn" :
                model, valid_loader = local_models[c], valid_loaders[c]
            else :
                model, valid_loader = server_model, valid_loaders[c]
            valid_loss, valid_cinc = naive_test(args, model, valid_loader, score_weights, sinus_rhythm_index, criterion, device=device)
            step_scores += valid_cinc

            print(f"{c} client valid loss : ", valid_loss)
            print(f"{c} client valid cinc score : ", valid_cinc)
            if wandb_run != None :
                wandb_run.log( {f'{c}_valid_loss': valid_loss, 'total_step' : total_steps} )
                wandb_run.log( {f'{c}_valid_cinc': valid_cinc, 'total_step' : total_steps} )


        step_scores = step_scores / len(args.data_list)
        if (step_scores > best_scores) or (comms == 0):
            patience = 0
            best_scores = step_scores
            best_models = update_best_models(args, best_models, local_models)
            save_models(args, best_models, SAVE_PATH, best=True)
            best_communication_rounds = comms + 1
        else :
            patience += 1

        save_models(args, local_models, SAVE_PATH, best=False)
    
    print("patience : ", patience)
    print("communication round of best model : ", best_communication_rounds)
    print("relative global epochs of best model : ", best_communication_rounds * args.local_epochs)

    if wandb_run :
        wandb_run.log( {'patience': patience} )
        wandb_run.log( {'communication_round_of_best_model': best_communication_rounds} )
        wandb_run.log( {'relative_global_epochs_of_best_model': best_communication_rounds * args.local_epochs} )

