# reference : https://github.com/mmcdermott/comprehensive_MTL_EHR/blob/3cea54aab5b14a8ed202dd39b095d081fc2b17c5/latent_patient_trajectories/representation_learner/adapted_model.py#L68

import torch, torch.optim, torch.nn as nn, torch.nn.functional as F
from torch.nn import TransformerEncoder
from collections import OrderedDict
import math

#################################################################################

task_dims = {
    'disch_24h': 10,
    'disch_48h': 10,
    'Final Acuity Outcome': 12,
    'tasks_binary_multilabel': 3, # Mort24, Mort48, Long Length of stay
    'next_timepoint': 15, 
    'next_timepoint_was_measured': 15,
    'masked_imputation': 15*2,
    'mort_24h' : 1, 
    'mort_48h' : 1, 
    'LOS' : 1 
    }

########################################################################################################

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    
    def __init__(self, task='disch_48h', norm_layer='gn', ntoken=128, d_model=256, nhead=8, d_hid=256, fc_layer_sizes = [256, 384],
                 nlayers=2, dropout=0.5): # d_model = 128 , d_hid=384
        super(TransformerModel, self).__init__()

        self.task = task

        self.ts_continuous_projector      = nn.Linear(165, d_model) # 16
        self.statics_continuous_projector = nn.Linear(15, d_model) # 16

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        if norm_layer == "gn" :
            encoder_layers = TransformerEncoderLayer(d_model, nhead, norm_layer="gn", dim_feedforward=d_hid, dropout=dropout) 
        else :
            encoder_layers = TransformerEncoderLayer(d_model, nhead, norm_layer="ln", dim_feedforward=d_hid, dropout=dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        output_dim = d_model
        # self.norm = nn.BatchNorm1d(output_dim)

        fc_stack = OrderedDict()
        for i, fc_layer_size in enumerate(fc_layer_sizes):
            fc_stack[f"fc_{i}"] = nn.Linear(output_dim, fc_layer_size)

            if norm_layer == "gn" :
                fc_stack[f"norm_{i}"] = nn.GroupNorm(2, fc_layer_size, eps=1e-5)
            elif norm_layer == "ln" :
                fc_stack[f"norm_{i}"] = nn.LayerNorm(fc_layer_size, eps=1e-5)
            elif norm_layer == "bn" :
                fc_stack[f"norm_{i}"] = nn.BatchNorm1d(fc_layer_size, eps=1e-5)
    
            fc_stack[f"relu_{i}"] = nn.ReLU()
            output_dim = fc_layer_size

        fc_stack["fc_last"] = nn.Linear(output_dim, 256)

        self.fc_stack = nn.Sequential(fc_stack)

        self.task_dim = task_dims[task]
        self.task_layer = nn.Linear(256, self.task_dim)


    def forward(self, ts_continuous, statics) :

        input_sequence     = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_continuous_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous

        output = self.pos_encoder(input_sequence)
        output = self.transformer_encoder(output)

        output = output.mean(dim=1) # avg

        # output = self.bn(output)
        output = self.fc_stack(output)
        output = self.task_layer(output)

        if self.task in ['tasks_binary_multilabel', 'mort_24h', 'mort_48h', 'LOS' ] :
            output = nn.Sigmoid()(output)

        return output

########################################################################################################

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, norm_layer="ln", dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) 

        self.linear1 = nn.Linear(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        
        if norm_layer == "gn" :
            self.norm1 = nn.GroupNorm(5, 25, eps=layer_norm_eps)
            self.norm2 = nn.GroupNorm(5, 25, eps=layer_norm_eps)
        else :
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps) 
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps) 

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

    def forward(self, src, src_mask, src_key_padding_mask) :

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x) :
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

####################################################################################################

