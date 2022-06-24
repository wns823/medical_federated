import json, pickle, enum, copy, os, random, sys, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Dict

############################################################################

class CachedDataset(Dataset): # Dataset
    def __init__(
        self,
        hospital_id,
        unitstays,
        data_path
    ):
        self.hospital_id = hospital_id
        self.unitstay = unitstays
        self.load_path = f"{data_path}/eicu-2.0/federated_preprocessed_data/cached_data"

    def __len__(self): 
        return len(self.unitstay)

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', None
                'ts', [batch_size, sequence_length, 165]
                'statics', [batch_size, 15]
                'next_timepoint',
                'next_timepoint_was_measured',
                'disch_24h', [batch_size, 10]
                'disch_48h', [batch_size, 10]
                'Final Acuity Outcome', [batch_size, 12]
                'ts_mask',
                'tasks_binary_multilabel', [batch_size, 3]
        """

        unitstay = self.unitstay[item]

        unitstay_path = os.path.join( self.load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)
        return tensors
