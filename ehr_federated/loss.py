import torch, torch.optim, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

class TaskBinaryMultilabelLoss(nn.Module):
    def __init__(self, binary_multilabel_loss_weight=None):
        super().__init__()
        # self.weights = binary_multilabel_loss_weight
        # self.weights.requires_grad_(False)
        params = {'pos_weight': binary_multilabel_loss_weight, 'reduction': 'none'}
        self.BCE_LL = nn.BCEWithLogitsLoss(**params)

    def forward(self, logits, labels):
        # new_weights = self.weights.unsqueeze(0).expand_as(logits)
        out = self.BCE_LL(logits, labels)
        # out = out * new_weights
        return out

def task_losses(task):
    if task == "disch_24h" or task == "disch_48h" :
        params = {'ignore_index': -1, 'reduction': 'none'}
        criterion = nn.CrossEntropyLoss(**params)
    elif task == 'Final Acuity Outcome' :
        params = {'ignore_index': -1}
        criterion = nn.CrossEntropyLoss(**params)
    elif task == 'tasks_binary_multilabel' :
        criterion = TaskBinaryMultilabelLoss()
    elif task == 'next_timepoint_was_measured' :
        params = {}
        criterion = nn.MultiLabelSoftMarginLoss(**params)
    elif task == 'mort_24h' or task == 'mort_48h' or task == 'LOS' :
        criterion = nn.BCELoss()
    else :
        criterion = nn.CrossEntropyLoss()
    
    return criterion

# if __name__ == "__main__" :
#     import pdb; pdb.set_trace()
#     TaskBinaryMultilabelLoss()