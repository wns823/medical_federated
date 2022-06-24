import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle, json, random
from PIL import Image
import torch

########################################################################

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])

test_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(), 
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])

########################################################################

class SkinCancerDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform is not None :
            self.transform = transform
        else :
            self.transform = transforms.ToTensor()

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, item):
        image_path = self.data[item]['img_path']

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(self.data[item]['extended_labels'].index(1.0))

        return image, label


def get_dataloader(args, total_data) :

    train_data = total_data['train']
    valid_data = total_data['valid']
    test_data = total_data['test']

    ################################################################################################

    train_dataset = SkinCancerDataset(train_data, transform=train_transform)
    valid_dataset = SkinCancerDataset(valid_data, transform=test_transform)
    test_dataset = SkinCancerDataset(test_data, transform=test_transform)

    if len(train_dataset) % args.batch_size == 1 :
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, drop_last=True)
    else :
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, drop_last=False)

    if len(valid_dataset) % args.batch_size == 1 :
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, drop_last=True)
    else :
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, drop_last=False)

    if len(test_dataset) % args.batch_size == 1 :
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, drop_last=True)
    else :
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloader

######################################################################################################

def get_federated_dataset(args):

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    print("### LOAD federated dataset ###")

    if "barcelona" in args.clients :
        barcelona_path = os.path.join( f"{args.data_path}/ISIC_2019", "ISIC_19_Barcelona_split.json")
        with open(barcelona_path, 'r') as f:
            barcelona_data = json.load(f)

        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, barcelona_data)

        client_weights.append( len(barcelona_data['train'] + barcelona_data['valid'] + barcelona_data['test'] ) ) # 0.7      
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "rosendahl" in args.clients :
        rosendahl_path = os.path.join( f"{args.data_path}/HAM10000", "HAM_rosendahl_split.json")
        with open(rosendahl_path, 'r') as f:
            rosendahl_data = json.load(f)

        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, rosendahl_data)

        client_weights.append( len(rosendahl_data['train'] + rosendahl_data['valid'] + rosendahl_data['test']  )) # 0.7
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "vienna" in args.clients :
        vienna_path = os.path.join( f"{args.data_path}/HAM10000", "HAM_vienna_split.json")
        with open(vienna_path, 'r') as f:
            vienna_data = json.load(f)

        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, vienna_data)

        client_weights.append( len(vienna_data['train'] + vienna_data['valid'] + vienna_data['test'] ) )
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "PAD_UFES_20" in args.clients :
        PAD_UFES_20_path = os.path.join(f"{args.data_path}/PAD-UFES-20", "PAD_UFES_20_split.json")
        with open(PAD_UFES_20_path, 'r') as f:
            PAD_UFES_20_data = json.load(f)

        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, PAD_UFES_20_data)

        client_weights.append( len(PAD_UFES_20_data['train'] + PAD_UFES_20_data['valid'] + PAD_UFES_20_data['test'] ) )
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "Derm7pt" in args.clients :
        Derm7pt_path = os.path.join(f"{args.data_path}/Derm7pt", "Derm7pt_split.json")
        with open(Derm7pt_path, 'r') as f:
            Derm7pt_data = json.load(f)

        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, Derm7pt_data)

        client_weights.append( len(Derm7pt_data['train'] + Derm7pt_data['valid'] + Derm7pt_data['test'] ) )
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    total_size = sum(client_weights)
    client_weights = [ float(c / total_size) for c in client_weights]

    return train_loaders, valid_loaders, test_loaders, client_weights


####################################################################################