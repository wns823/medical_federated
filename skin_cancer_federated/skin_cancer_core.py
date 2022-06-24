import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import preprocessing
from torch.nn.functional import normalize
import numpy as np

def train_single(model, train_loader, optimizer, criterion, device='cpu'):
    epoch_loss = 0

    model.train()
    for img, label in train_loader :
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device).long()
        prediction = model(img)

        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        torch.cuda.empty_cache()

    epoch_loss = float(epoch_loss)/ len(train_loader)

    return epoch_loss


def train_fedprox(args, model, server_model, train_loader, optimizer, criterion, param_name, device='cpu'):
    epoch_loss = 0

    model.train()
    for idx, data in enumerate(train_loader) :
        optimizer.zero_grad()
        img, label = data[0].to(device), data[1].to(device).long()
        prediction = model(img)

        loss = criterion(prediction, label)

        if idx > 0:
            w_diff = torch.tensor(0., device=device)
            for param, w, w_t in zip(param_name, server_model.parameters(), model.parameters()):
                if args.algorithm != "fedproxn" :
                    w_diff += torch.pow(torch.norm(w - w_t), 2)
                else :
                    if ("gn" not in param) and ("bn" not in param) :
                        w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = float(epoch_loss)/ len(train_loader)

    return epoch_loss


def train_feddyn(args, model, train_loader, optimizer, criterion, server_model_param, prev_grads, device='cpu'):
    epoch_loss = 0

    model.train()
    for img, label in train_loader :
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device).long()
        prediction = model(img)

        loss = criterion(prediction, label)

        ######
        ## Dynamic regularizer
                
        local_par_list = None
        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
            # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                                
        loss_algo = args.feddyn_alpha * torch.sum(local_par_list * (-server_model_param + prev_grads))
        loss += loss_algo


        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = float(epoch_loss)/ len(train_loader)

    return epoch_loss

#############################################################################################################

def test_single(model, test_loader, criterion, device='cpu'):
    epoch_loss = 0
    predictions = []
    labels = []
    label_list = []
    pred_scores = []

    model.eval()
    with torch.no_grad():
        for img, label in test_loader :
            img, label = img.to(device), label.to(device).long()
            prediction = model(img)
            
            # pred_scores.append( torch.nn.functional.softmax(prediction, dim=-1).cpu().detach().numpy() )
            pred_scores.extend( prediction.cpu().detach().numpy() )

            label_list.extend( ((label.cpu()).numpy()).tolist() )

            loss = criterion(prediction, label)
            epoch_loss += loss.item()

            y_true = label.detach().cpu().numpy()
            
            labels.extend(y_true)

    epoch_loss = float(epoch_loss)/ len(test_loader)

    pred_scores = np.array(pred_scores)
    pred_scores = pred_scores[:,list(set(label_list))]
    predictions = np.argmax(pred_scores, axis=1)
    pred_scores = torch.nn.functional.softmax( torch.tensor(pred_scores), dim=-1 ).numpy()

    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)

    # f1 = 100 * f1_score(y_true=labels, y_pred=predictions, average='macro')
    # reference : https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
    pr_auc  = 100.0 * average_precision_score(lb.transform(labels), lb.transform(predictions), average='macro')
    roc_auc = 100.0 * roc_auc_score(y_true=np.array(label_list), y_score=pred_scores, multi_class='ovr', average='macro')

    return epoch_loss, pr_auc, roc_auc

