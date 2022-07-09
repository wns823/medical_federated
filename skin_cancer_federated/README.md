# Federated learning with skin cancer images

## Requirements
* [PyTorch](https://pytorch.org) version >= 1.9.0
* Python version >= 3.8
* For training new models, you'll also need an NIVDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

```bash
$ conda create -n medical_image python=3.8 (optional)
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install wandb
$ pip install efficientnet-pytorch==0.7.1
$ pip install pandas==1.4.2
$ pip install scikit-learn==1.1.1
```

## Prepare dataset
- Make the folder to save datasets
```bash
$ mkdir [your specific path]/data_storage && cd [your specific path]/data_storage
```

### Download dataset
- Download the ISIC-19 dataset in the data_storage folder
```bash
$ wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
$ wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
$ wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv
$ bash scripts/ISIC19.sh [data_storage_path]
```

- Download the HAM10000 dataset from [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) in the data_storage folder
```bash
$ bash scripts/HAM10000.sh [data_storage_path]
```

- Download the PAD-UFES-20 dataset from [link](https://data.mendeley.com/datasets/zr7vgbcyr2/1) in the data_storage folder
```bash
$ bash scripts/PAD_UFES.sh [data_storage_path]
```

- Download the Derm7pt dataset from [link](https://derm.cs.sfu.ca/Welcome.html) in the data_storage folder
```bash
$ bash scripts/Derm7pt.sh [data_storage_path]
```

### Preprocess dataset
- Run the following script to pre-process skin images 

```bash
python preprocess_skin.py --data_path [data_storage_path]
python split_dataset.py --data_path [data_storage_path]
```

## Train

For fedavg/fedbn:
```bash
python skin_cancer_main.py --algorithm "fedavg" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --learning_rate 0.001 --data_path [data_storage_path]
```

For fedprox/fedpxn:
```bash
python skin_cancer_main.py --algorithm "fedprox" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --mu 0.01 --learning_rate 0.001 --data_path [data_storage_path]
```


For fedopt(fedadam/fedadagrad/fedyogi):
```bash
python skin_cancer_main.py --algorithm "fedadam" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --server_learning_rate 0.01 --learning_rate 0.001 --tau 0.01 \
    --data_path [data_storage_path]
```


For feddyn:
```bash
python skin_cancer_main.py --algorithm "feddyn" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --feddyn_alpha 0.01 --learning_rate 0.001 --data_path [data_storage_path]
```


For efficientnet with group normalization:
```bash
--model_type "efficientnet_gn"
```

## Test

For fedavg/fedbn:
```bash
python skin_cancer_main.py --algorithm "fedavg" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --learning_rate 0.001 --data_path [data_storage_path] --test
```

For fedprox/fedpxn:
```bash
python skin_cancer_main.py --algorithm "fedprox" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --mu 0.01 --learning_rate 0.001 --data_path [data_storage_path] --test
```


For fedopt(fedadam/fedadagrad/fedyogi):
```bash
python skin_cancer_main.py --algorithm "fedadam" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --server_learning_rate 0.01 --learning_rate 0.001 --tau 0.01 \
    --data_path [data_storage_path] --test
```


For feddyn:
```bash
python skin_cancer_main.py --algorithm "feddyn" --communication 300 \
    --local_epochs 1 --model_type "efficientnet" --client "barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt" \
    --seed 0 --feddyn_alpha 0.01 --learning_rate 0.001 --data_path [data_storage_path] --test
```