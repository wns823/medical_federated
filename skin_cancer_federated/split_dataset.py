# Split train/valid/test
import os, json, argparse, random

def add_fit_args(parser):
    parser.add_argument('--data_path', type=str, default='data_storage', help="Data path")
    parser.add_argument('--seed', type=int, default=0, help="Seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = add_fit_args(argparse.ArgumentParser())
    random.seed(args.seed)

    common_path = args.data_path

    # Barcelona
    ###############################################################
    barcelona_path = os.path.join( f"{common_path}/ISIC_2019", "ISIC_19_Barcelona.json") # "ISIC_19_Barcelona_split.json
    with open(barcelona_path, 'r') as f:
        barcelona_data = json.load(f)['data']

    random.shuffle(barcelona_data)
    total_size = len(barcelona_data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    barcelona_split = {'train' : barcelona_data[:train_size], 'valid' : barcelona_data[train_size:train_size+valid_size], 'test' : barcelona_data[train_size+valid_size:]}

    file_path = os.path.join( f"{common_path}/ISIC_2019", "ISIC_19_Barcelona_split.json")
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(barcelona_split, file, indent="\t")

    # Rosendahl
    ###############################################################
    rosendahl_path = os.path.join( f"{common_path}/HAM10000", "HAM_rosendahl.json") # HAM_rosendahl_split.json
    with open(rosendahl_path, 'r') as f:
        rosendahl_data = json.load(f)['data']

    random.shuffle(rosendahl_data)
    total_size = len(rosendahl_data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    rosendahl_split = {'train' : rosendahl_data[:train_size], 'valid' : rosendahl_data[train_size:train_size+valid_size], 'test' : rosendahl_data[train_size+valid_size:]}

    file_path = os.path.join( f"{common_path}/HAM10000", "HAM_rosendahl_split.json")
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(rosendahl_split, file, indent="\t")

    # Vienna
    ###############################################################
    vienna_path = os.path.join( f"{common_path}/HAM10000", "HAM_vienna.json") # HAM_vienna_split.json
    with open(vienna_path, 'r') as f:
        vienna_data = json.load(f)['data']

    random.shuffle(vienna_data)
    total_size = len(vienna_data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    vienna_split = {'train' : vienna_data[:train_size], 'valid' : vienna_data[train_size:train_size+valid_size], 'test' : vienna_data[train_size+valid_size:]}

    file_path = os.path.join( f"{common_path}/HAM10000", "HAM_vienna_split.json")
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(vienna_split, file, indent="\t")


    # UFES
    ###############################################################
    PAD_UFES_20_path = os.path.join( f"{common_path}/PAD-UFES-20", "PAD_UFES_20.json") # PAD_UFES_20_split.json
    with open(PAD_UFES_20_path, 'r') as f:
        PAD_UFES_20_data = json.load(f)['data']

    random.shuffle(PAD_UFES_20_data)
    total_size = len(PAD_UFES_20_data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    PAD_UFES_20_split = {'train' : PAD_UFES_20_data[:train_size], 'valid' : PAD_UFES_20_data[train_size:train_size+valid_size], 'test' : PAD_UFES_20_data[train_size+valid_size:]}

    file_path = os.path.join( f"{common_path}/PAD-UFES-20", "PAD_UFES_20_split.json")
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(PAD_UFES_20_split, file, indent="\t")


    # Canada
    ###############################################################
    Derm7pt_path = os.path.join( f"{common_path}/Derm7pt", "Derm7pt.json") # Derm7pt_split.json
    with open(Derm7pt_path, 'r') as f:
        Derm7pt_data = json.load(f)['data']


    random.shuffle(Derm7pt_data)
    total_size = len(Derm7pt_data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    Derm7pt_split = {'train' : Derm7pt_data[:train_size], 'valid' : Derm7pt_data[train_size:train_size+valid_size], 'test' : Derm7pt_data[train_size+valid_size:]}

    file_path = os.path.join( f"{common_path}/Derm7pt", "Derm7pt_split.json") 
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(Derm7pt_split, file, indent="\t")
