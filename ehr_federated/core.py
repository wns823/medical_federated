import os, torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import torch.nn.functional as F

def check_labels(args, train_patients, valid_patients, test_patients):
    load_path = "/home/data_storage/eicu-2.0/federated_preprocessed_data/cached_data"
    binary_task = ['mort_24h', 'mort_48h', 'LOS' ]
    idx = binary_task.index(args.task)

    train_labels = []
    for unitstay in train_patients:
        unitstay_path = os.path.join( load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)
        value = tensors['tasks_binary_multilabel'][idx].item()
        if np.isnan(value) :
            pass
        else :
            train_labels.append( tensors['tasks_binary_multilabel'][idx].item() )

    valid_labels = []
    for unitstay in valid_patients:
        unitstay_path = os.path.join( load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)
        value = tensors['tasks_binary_multilabel'][idx].item()
        if np.isnan(value) :
            pass
        else :
            valid_labels.append( tensors['tasks_binary_multilabel'][idx].item() )

    test_labels = []
    for unitstay in test_patients:
        unitstay_path = os.path.join( load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)
        value = tensors['tasks_binary_multilabel'][idx].item()
        if np.isnan(value) :
            pass
        else :
            test_labels.append( tensors['tasks_binary_multilabel'][idx].item() )

    if len(set(train_labels)) == 1 or len(set(valid_labels)) == 1 or len(set(test_labels)) == 1 :
        return False
    else :
        return True


def train_naive(args, f_model, train_dataloader, criterion, optimizer, device='cpu', scheduler=None):
    f_model.train()

    total_loss = 0.0

    for idx, batch in enumerate(train_dataloader) :
        for k in (
            'ts', 'statics', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
            'ts_vals', 'ts_is_measured', 'ts_mask',
            ):
            if k in batch:
                batch[k] = batch[k].to(device).float()
                # batch[k] = batch[k].float().to(device)


        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_ftseq'):
            if k in batch: 
                # batch[k] = batch[k].to(device).squeeze().long()
                batch[k] = batch[k].squeeze().long()

        optimizer.zero_grad()

        ts, statics  = batch['ts'], batch['statics']

        logit = f_model(ts, statics)

        if args.task in ['disch_24h', 'disch_48h' ] :
            labels = batch[args.task].to(device)
            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)

            isnan = (labels == -9223372036854775808).to(device)
            labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
            loss = torch.where(isnan, torch.zeros_like(labels, dtype=torch.float32).to(device), criterion(logit, labels_smoothed) )
            loss = loss.mean()
        elif args.task in ['mort_24h', 'mort_48h', 'LOS' ] :
            t = ['mort_24h', 'mort_48h', 'LOS' ].index(args.task)
            labels = batch['tasks_binary_multilabel'][:,t].unsqueeze(1)#.to(device)

            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)

            isnan = torch.isnan(labels).to(device)
            labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
            loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
            loss = loss.mean()

        else :
            labels = batch[args.task].to(device)
            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)
            loss = criterion(logit, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # torch.cuda.empty_cache()

    if scheduler is not None :
        scheduler.step()

    return float(total_loss) / len(train_dataloader)

############################################################################################################

def train_fedprox(args, global_model, f_model, train_dataloader, criterion, optimizer, param_name, device='cpu', scheduler=None):
    f_model.train()

    total_loss = 0.0

    for idx, batch in enumerate(train_dataloader) :
        for k in (
            'ts', 'statics', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
            'ts_vals', 'ts_is_measured', 'ts_mask',
            ):
            if k in batch:
                batch[k] = batch[k].to(device).float()
        
        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_ftseq'):
            if k in batch: 
                batch[k] = batch[k].squeeze().long()

        optimizer.zero_grad()

        ts, statics  = batch['ts'], batch['statics']
        logit = f_model(ts, statics)

        if args.task in ['disch_24h', 'disch_48h' ] :
            labels = batch[args.task].to(device)
            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)

            isnan = (labels == -9223372036854775808).to(device)
            labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
            loss = torch.where(isnan, torch.zeros_like(labels, dtype=torch.float32).to(device), criterion(logit, labels_smoothed) )
            loss = loss.mean()
        elif args.task in ['mort_24h', 'mort_48h', 'LOS' ] :
            t = ['mort_24h', 'mort_48h', 'LOS' ].index(args.task)
            labels = batch['tasks_binary_multilabel'][:,t].unsqueeze(1).to(device)
            isnan = torch.isnan(labels).to(device)
            labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
            loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
            loss = loss.mean()            
        else :
            labels = batch[args.task].to(device)
            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)
            loss = criterion(logit, labels)

        if idx > 0:
            w_diff = torch.tensor(0., device=device)
            for name, w, w_t in zip(param_name, global_model.parameters(), f_model.parameters()):
                if args.algorithm == "fedproxn":
                    if "norm" not in name :
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                else :
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

            # w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # torch.cuda.empty_cache()

    if scheduler is not None :
        scheduler.step()

    return float(total_loss) / len(train_dataloader)


def test_naive(args, f_model, test_dataloader, criterion, device='cpu') :
    f_model.eval()

    binary_task = ['mort_24h', 'mort_48h', 'LOS']
    total_pred, total_label, total_score = [], [], []
    AUROC_macro, AUPRC_macro = 0.0, 0.0
    total_loss = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader) :
            for k in (
                'ts', 'statics', 'next_timepoint_was_measured', 'next_timepoint',
                'tasks_binary_multilabel',
                'ts_vals', 'ts_is_measured', 'ts_mask',
                ):
                if k in batch:
                    batch[k] = batch[k].to(device).float()
                    # batch[k] = batch[k].float().to(device)
            
            for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_ftseq'):
                if k in batch: 
                    batch[k] = batch[k].squeeze().long()

            ts, statics  = batch['ts'], batch['statics']
            logit = f_model(ts, statics)

            if args.task in ['disch_24h', 'disch_48h' ] :
                labels = batch[args.task].to(device)
                if labels.dim() == 0 :
                    labels = labels.unsqueeze(0)

                isnan = (labels == -9223372036854775808).to(device)

                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
                loss = torch.where(isnan, torch.zeros_like(labels, dtype=torch.float32).to(device), criterion(logit, labels_smoothed) )
                loss = loss.mean()
                
                _, pred = logit.max(1)
                pred = pred.cpu()

                total_score.extend( logit.detach().cpu().numpy() )
                total_pred.extend( pred.numpy() )
                total_label.extend( labels.detach().cpu().numpy() )

            elif args.task in binary_task :
                t = binary_task.index(args.task)
                labels = batch['tasks_binary_multilabel'][:,t].unsqueeze(1).to(device)
                if labels.dim() == 0 :
                    labels = labels.unsqueeze(0)

                isnan = torch.isnan(labels).to(device)

                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
                loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
                loss = loss.mean()

                pred = logit.detach().cpu().numpy()
                target = labels.detach().cpu().numpy()

                total_pred.extend( pred )
                total_label.extend( target )

            else :
                labels = batch[args.task].to(device)
                if labels.dim() == 0 :
                    labels = labels.unsqueeze(0)

                loss = criterion(logit, labels)

                _, pred = logit.max(1)
                pred = pred.cpu()

                total_score.extend( logit.detach().cpu().numpy() )
                total_pred.extend( pred.numpy() )
                total_label.extend( labels.detach().cpu().numpy() )

            total_loss += loss.item()

            # torch.cuda.empty_cache()

    test_loss = float(total_loss) / len(test_dataloader)

    if args.task in binary_task :
        mask = np.isnan(total_label)
        post_label = np.array(total_label).astype(int)[~mask]
        post_pred = np.array(total_pred)[~mask]

        Accuracy = accuracy_score(post_label, np.where(post_pred < 0.5, 0, 1)) * 100        
        AUROC_macro = roc_auc_score( post_label, post_pred, average='macro' )
        AUPRC_macro = average_precision_score( post_label, post_pred, average='macro' )           
        return test_loss , Accuracy, AUROC_macro, AUPRC_macro

    else :
        mask = (np.array(total_label) == -9223372036854775808) # ['disch_24h', 'disch_48h' ]

        post_label = np.array(total_label).astype(int)[~mask]
        post_pred = np.array(total_pred)[~mask]
        post_score = np.array(total_score)[~mask]

        ########################################################################
        
        lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        lb.fit(  list(range(post_score.shape[1])) )
        true_binarized = lb.transform(post_label)

        label_cnts = np.array([len(set(true_binarized[:, i])) for i in range(true_binarized.shape[1])])
        valid_labels = (label_cnts > 1)

        true_binarized = true_binarized[:, valid_labels]
        score_binarized = np.nan_to_num(np.round(post_score[:, valid_labels], 6))

        ###
        auroc_macro = roc_auc_score(true_binarized, score_binarized, average='macro') # AUROC (ovr; macro)
        auprc_macro = average_precision_score(true_binarized, score_binarized, average='macro' )           

        ########################################################################
        Accuracy = accuracy_score(post_label, post_pred) * 100

        return test_loss, Accuracy, auroc_macro, auprc_macro

######################################################################

def train_feddyn(args, f_model, server_model_param, train_dataloader, criterion, optimizer, prev_grads, device='cpu', scheduler=None):
    f_model.train()

    total_loss = 0.0

    for idx, batch in enumerate(train_dataloader) :
        for k in (
            'ts', 'statics', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
            'ts_vals', 'ts_is_measured', 'ts_mask',
            ):
            if k in batch:
                batch[k] = batch[k].to(device).float()

        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_ftseq'):
            if k in batch: 
                batch[k] = batch[k].squeeze().long()

        optimizer.zero_grad()

        ts, statics  = batch['ts'], batch['statics']

        logit = f_model(ts, statics)

        if args.task in ['disch_24h', 'disch_48h' ] :
            labels = batch[args.task].to(device)
            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)

            isnan = (labels == -9223372036854775808).to(device)
            labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
            loss = torch.where(isnan, torch.zeros_like(labels, dtype=torch.float32).to(device), criterion(logit, labels_smoothed) )
            loss = loss.mean()
        elif args.task in ['mort_24h', 'mort_48h', 'LOS' ] :
            t = ['mort_24h', 'mort_48h', 'LOS' ].index(args.task)
            labels = batch['tasks_binary_multilabel'][:,t].unsqueeze(1)#.to(device)

            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)

            isnan = torch.isnan(labels).to(device)
            labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
            loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
            loss = loss.mean()

        else :
            labels = batch[args.task].to(device)
            if labels.dim() == 0 :
                labels = labels.unsqueeze(0)
            loss = criterion(logit, labels)

        ######
        ## Dynamic regularizer
                
        local_par_list = None
        for param in f_model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
            # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                                
        loss_algo = args.feddyn_alpha * torch.sum(local_par_list * (-server_model_param + prev_grads))
        loss += loss_algo
        ######
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=f_model.parameters(), max_norm=10)
        optimizer.step()
        total_loss += loss.item()

    if scheduler is not None :
        scheduler.step()
        
    return float(total_loss) / len(train_dataloader)