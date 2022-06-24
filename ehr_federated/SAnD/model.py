import torch
import torch.nn as nn
from SAnD import modules
import torch.nn.functional as F

# reference : https://github.com/khirotaka/SAnD

task_dims = {
    'disch_24h': 10,
    'disch_48h': 10,
    'Final Acuity Outcome': 12,
    'tasks_binary_multilabel': 3,
    'next_timepoint': 15, 
    'next_timepoint_was_measured': 15,
    'masked_imputation': 15*2,
    'mort_24h' : 1, 
    'mort_48h' : 1, 
    'LOS' : 1
    }

# in_feature = 23
# seq_len = 256
# n_heads = 32
# factor = 32
# num_class = 10
# num_layers = 6

class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model, dropout_rate=0.2):
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)

        self.bn = nn.BatchNorm1d(self.d_model)

        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)

        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = F.relu(self.bn(x))
        x = x.transpose(1, 2)


        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x

class SAnD(nn.Module):
    def __init__(self, task, input_features=128, seq_len=25, n_heads=8, factor=10, n_layers=1, d_model=256, dropout_rate=0.2) :
        super(SAnD, self).__init__()

        self.task = task
        self.ts_continuous_projector      = nn.Linear(165, input_features) # 16
        self.statics_continuous_projector = nn.Linear(15, input_features) # 16

        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        n_class = task_dims[task]
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, ts_continuous, statics, device='cpu') :

        input_sequence     = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_continuous_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)
        input_sequence += statics_continuous # torch.Size([64, 25, 128])

        outputs = self.encoder(input_sequence)
        outputs = self.dense_interpolation(outputs)
        outputs = self.clf(outputs)

        if self.task in ['tasks_binary_multilabel', 'mort_24h', 'mort_48h', 'LOS' ] :
            outputs = nn.Sigmoid()(outputs)

        return outputs
