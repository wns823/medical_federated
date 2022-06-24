import os, wandb, json, torch
from cached_dataset import *
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from base_model import *
from loss import *
from core import *
from communication_func import *
from copy import deepcopy
from SAnD.model import SAnD


def nan_checker( args, patients ):
    
    new_patients = []

    for patient in patients :
        temp = CachedDataset("total", [patient], args.data_path)

        if args.task == "mort_24h" or args.task == "mort_48h" or args.task == "LOS" :
            index = ["mort_24h", "mort_48h", "LOS"].index(args.task)
            label = temp[0]['tasks_binary_multilabel'][index]
        else :
            label = temp[0][args.task]

        check = torch.isnan(label).item()

        if check == False :
            new_patients.append(patient)

    return new_patients


def get_dataset(args):
    new_dir = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/data_split_fixed"
    client_id = args.hospital_id
    
    train_dataset, valid_dataset, test_dataset = [], [], []

    for c_id in client_id :
        each_client = str(c_id)

        ##########################################################
        if args.task in ['mort_24h', 'mort_48h', 'LOS' ] :
            with open( os.path.join(new_dir, f"{c_id}.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']
        else :
            with open( os.path.join(new_dir, f"{c_id}_ver2.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']

        ##########################################################

        print(f"{c_id} client's patient number : ", len(train_patients + valid_patients + test_patients))

        train_patients = nan_checker( args, train_patients )
        valid_patients = nan_checker( args, valid_patients )
        test_patients = nan_checker( args, test_patients )

        train_dataset.append( CachedDataset( each_client, train_patients, args.data_path) )
        valid_dataset.append( CachedDataset( each_client, valid_patients, args.data_path) )
        test_dataset.append( CachedDataset( each_client, test_patients, args.data_path) )

    return client_id, train_dataset, valid_dataset, test_dataset


def get_dataloader(args, train_dataset, valid_dataset, test_dataset):

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    for i in range(len(args.hospital_id)) :
        client_weights.append( len(train_dataset[i]) )

        train_sampler = RandomSampler(train_dataset[i])
        if len(train_dataset[i]) % args.batch_size == 1:
            train_loaders.append( DataLoader(train_dataset[i], sampler=train_sampler, batch_size=args.batch_size, num_workers=8, drop_last=True) )
        else :
            train_loaders.append( DataLoader(train_dataset[i], sampler=train_sampler, batch_size=args.batch_size, num_workers=8, drop_last=False) )

        valid_sampler = RandomSampler(valid_dataset[i])
        if len(valid_dataset[i]) % args.batch_size == 1 :
            valid_loaders.append( DataLoader(valid_dataset[i], sampler=valid_sampler, batch_size=args.batch_size, num_workers=8, drop_last=True) )
        else :
            valid_loaders.append( DataLoader(valid_dataset[i], sampler=valid_sampler, batch_size=args.batch_size, num_workers=8, drop_last=False) )

        if len(test_dataset[i]) % args.batch_size == 1 :
            test_loaders.append( DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=8, drop_last=True) )
        else :
            test_loaders.append( DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=8, drop_last=False) )


    total = sum(client_weights)
    client_weights = [ weight / total for weight in client_weights ]

    return client_weights, train_loaders, valid_loaders, test_loaders


def update_best_models(args, best_models, models) :
    for idx in range(len(args.hospital_id)) :
        best_models[idx] = deepcopy(models[idx])
    return best_models

def save_models(args, models, task, save_path, best=False):
    model_list = {}
    
    if args.task == 'Final Acuity Outcome' :
        task_name = 'Final_Acuity_Outcome'
    else :
        task_name = args.task


    if best == True :
        for idx in range(len(args.hospital_id)) :
            model_name = 'model_' + str(idx)
            model_list[model_name] = models[idx].state_dict()
        
        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_best_{task_name}_{args.model_type}_{args.learning_rate}_{args.server_learning_rate}_{args.tau}_{args.seed}.pt" ) )    
        else :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_best_{task_name}_{args.model_type}_{args.seed}.pt" ) )
        
    else :
        for idx in range(len(args.hospital_id)) :
            model_name = 'model_' + str(idx)
            model_list[model_name] = models[idx].state_dict()

        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_{task_name}_{args.model_type}_{args.learning_rate}_{args.server_learning_rate}_{args.tau}_{args.seed}.pt" ) )    
        else :
            torch.save( model_list , os.path.join( save_path, f"checkpoint_{task_name}_{args.model_type}_{args.seed}.pt" ) )
        


def initialization(args, model):
    factor = deepcopy(model)

    for key in factor.state_dict().keys():
        if 'num_batches_tracked' in key:
            pass;
        else:
            temp = torch.zeros_like(factor.state_dict()[key])
            # temp.new_full(temp.shape, args.tau ** 2)
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

    prev_grads = [deepcopy(factor) for idx in range(len(args.hospital_id))]

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

############################################################################################################

def ehr_federated_main(args, wandb_run=None) :

    naming = "max_" + str(len(args.hospital_id))

    SAVE_PATH = f'{args.save_path}/federated/{args.algorithm}_{args.communications}_{args.local_epochs}/{naming}'

    if not os.path.exists( SAVE_PATH ):
        os.makedirs( SAVE_PATH )

    patience = 0
    best_communication_rounds = 0

    client_id, train_dataset, valid_dataset, test_dataset = get_dataset(args)

    client_weights, train_loaders, valid_loaders, test_loaders = get_dataloader(args, train_dataset, valid_dataset, test_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model_type == "transformer_gn" :
        server_model = TransformerModel(task=args.task, norm_layer='gn').to(device)
    elif args.model_type == "transformer_ln" :
        server_model = TransformerModel(task=args.task, norm_layer='ln').to(device)
    else :
        server_model = TransformerModel(task=args.task, norm_layer='x').to(device)

    param_name = []
    for name, param in server_model.named_parameters() :
        param_name.append(name)

    models = [deepcopy(server_model).to(device) for idx in range(len(args.hospital_id))]

    if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
        m_t = initialization(args, server_model)
        v_t = initialization(args, server_model)

    if args.algorithm == "feddyn" :
        prev_grads = initialization_prev_grad(args, server_model)
        cld_model = deepcopy(server_model)

    criterion = task_losses(args.task)
    best_scores = 0.0
    best_models = []
    for idx in range(len(args.hospital_id)):
        best_models.append( deepcopy(server_model) )

    if args.test :
        if args.task == 'Final Acuity Outcome' :
            task_name = 'Final_Acuity_Outcome'
        else :
            task_name = args.task

        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            checkpoint = torch.load( os.path.join(f'{SAVE_PATH}', f'checkpoint_best_{task_name}_{args.model_type}_{args.learning_rate}_{args.server_learning_rate}_{args.tau}_{args.seed}.pt'))
        else :
            checkpoint = torch.load( os.path.join(f'{SAVE_PATH}', f'checkpoint_best_{task_name}_{args.model_type}_{args.seed}.pt'))
        
        for client_idx in range(len(args.hospital_id)):
            models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])

        print(f"communications : {args.communications}, local epochs : {args.local_epochs}, {args.algorithm} {args.task} results")
        for c in range(len(args.hospital_id)) :
            model, test_loader = models[c], test_loaders[c]
            print(f"########## {c} client's test result ##########")
            test_loss, test_accuracy, auroc_macro, auprc_macro = test_naive(args, model, test_loader, criterion, device=device)

            print(f"{c} client test loss : ", test_loss)
            print(f"{c} client test accuracy : ", test_accuracy)
            print(f"{c} client test auroc(macro) : ", auroc_macro)
            print(f"{c} client test auprc(macro) : ", auprc_macro)
            print()

        exit()


    for comms in range(args.communications):

        if args.algorithm != "feddyn" :
            optimizers = [ torch.optim.Adam( models[idx].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) for idx in range(len(args.hospital_id)) ]
        else :
            optimizers = [ torch.optim.Adam( models[idx].parameters(), lr=args.learning_rate, weight_decay=args.feddyn_alpha + args.weight_decay) for idx in range(len(args.hospital_id)) ]

        schedulers = [ torch.optim.lr_scheduler.StepLR(optimizers[idx], args.learning_rate_step, args.learning_rate_decay) for idx in range(len(args.hospital_id)) ]

        for c in range(len(args.hospital_id)) :
            model, train_loader, optimizer, scheduler = models[c], train_loaders[c], optimizers[c], schedulers[c]

            if args.algorithm == "feddyn" :
                prev_grad_tensor = get_param_tensor(prev_grads[c])
                server_model_tensor = get_param_tensor(server_model)

            for epoch in range(args.local_epochs) :
                total_steps = epoch + comms * args.local_epochs

                if (args.algorithm == "fedprox" or args.algorithm == "fedpxn" ) and comms > 0 :
                    train_loss = train_fedprox(args, server_model, model, train_loader, criterion, optimizer, param_name, device=device, scheduler=scheduler)
                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )
                elif args.algorithm == "feddyn" :
                    train_loss = train_feddyn(args, model, server_model_tensor, train_loader, criterion, optimizer, prev_grad_tensor, device=device, scheduler=scheduler)
                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )
                else :
                    train_loss = train_naive(args, model, train_loader, criterion, optimizer, device=device, scheduler=scheduler)
                    print(f"[{c} client {total_steps} steps] train loss : ", train_loss)
                    if wandb_run :
                        wandb_run.log( {f'{c}_train_loss ': train_loss, 'total_step' : total_steps} )

            if args.algorithm == "feddyn" :
                prev_grads[c] = get_grad_diff( prev_grads[c], model, cld_model)

        ###################################################################################################
        if args.algorithm == "fedadam" or args.algorithm == "fedadagrad" or args.algorithm == "fedyogi" :
            server_model, models, m_t, v_t = communication_fedopt(args, server_model, models, client_weights, m_t, v_t)
        elif args.algorithm == "feddyn" :
            server_model, models, cld_model = communication_FedDyn(args, server_model, models, client_weights, cld_model, prev_grads)
        else :
            server_model, models = communication(args, server_model, models, client_weights)
        ###################################################################################################

        total_steps = (comms + 1) * args.local_epochs
        step_scores = 0.0

        for c in range(len(args.hospital_id)) :
            if args.algorithm != "feddyn" :
                model, valid_loader = models[c], valid_loaders[c]
            else :
                model, valid_loader = server_model, valid_loaders[c]

            valid_loss, valid_accuracy, auroc_macro, auprc_macro = test_naive(args, model, valid_loader, criterion, device=device)
            step_scores += auroc_macro
            print(f"{c} client valid loss : ", valid_loss)
            print(f"{c} client valid accuracy : ", valid_accuracy)
            print(f"{c} client valid AUROC(macro) : ", auroc_macro)
            print(f"{c} client valid AUPRC(macro) : ", auprc_macro)

            if wandb_run :
                wandb_run.log( {f'{c}_valid_loss ': valid_loss, 'total_step' : total_steps} )
                wandb_run.log( {f'{c}_valid_accuracy ': valid_accuracy, 'total_step' : total_steps} )
                wandb_run.log( {f'{c}_valid_AUROC(macro) ': auroc_macro, 'total_step' : total_steps} )
                wandb_run.log( {f'{c}_valid_AUPRC(macro) ': auprc_macro, 'total_step' : total_steps} )

        # save best model
        step_scores = step_scores / len(args.hospital_id)
        if step_scores > best_scores :
            patience = 0
            best_scores = step_scores
            best_models = update_best_models(args, best_models, models)
            save_models(args, best_models, args.task, SAVE_PATH, best=True)
            best_communication_rounds = comms + 1
        else :
            patience += 1

        save_models(args, models, args.task, SAVE_PATH, best=False)
    
    print("patience : ", patience)
    print("communication round of best model : ", best_communication_rounds)
    print("relative global epochs of best model : ", best_communication_rounds * args.local_epochs)

    if wandb_run :
        wandb_run.log( {'patience': patience} )
        wandb_run.log( {'communication_round_of_best_model': best_communication_rounds} )
        wandb_run.log( {'relative_global_epochs_of_best_model': best_communication_rounds * args.local_epochs} )

