# Federated learning (FL) with ECG data

For easy ECG data processing, the code of FL with ECG data is based on the [Fairseq-signals](https://github.com/Jwoo5/fairseq-signals)

# Requirements and Installation
* [PyTorch](https://pytorch.org) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NIVDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-signals** from source and develop locally:

```bash
$ conda create -n ecg_federated python=3.8 (optional)
$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install --editable ./
$ pip install torch-ecg==0.0.15
$ pip install easydict
$ pip install wandb
$ python setup.py build_ext --inplace
```

## Prepare ECG dataset

- Make the folder to save datasets
```bash
$ mkdir [your specific path]/data_storage && cd [your specific path]/data_storage
```

- Download ECG dataset in the data_storage folder
```shell script
$ bash scripts/download_datasets.sh
```
All datasets are originated from [PhysioNet2021](https://moody-challenge.physionet.org/2021/).

### Pre-process

```shell script
$ bash scripts/preprocess.sh [data_storage_path]
$ bash scripts/get_manifest.sh [data_storage_path]
```

## Train

For fedavg/fedbn :
```bash
python main.py --communications 200 --local_epochs 1 \
    --model_type "resnet" --learning_rate 0.001 --algorithm "fedavg" --seed 0 \
    --load_dir "/home/edlab/sjyang/temp/data_storage/federated_ecg_manifest"
```

For fedprox/fedpxn :
```bash
python main.py --communications 200 --local_epochs 1 \
    --model_type "resnet" --learning_rate 0.001 --algorithm "fedprox" --mu 0.01 --seed 0 \
    --load_dir "/home/edlab/sjyang/temp/data_storage/federated_ecg_manifest"    
 ```

For fedadam :
```bash
python main.py --communications 200 --local_epochs 1 \
    --model_type "resnet" --learning_rate 0.001 --algorithm "fedadam" --seed 0 \
    --server_learning_rate 0.01 --tau 0.01 \
    --load_dir "/home/edlab/sjyang/temp/data_storage/federated_ecg_manifest"  
```

For ResNet-NC-SE with group normalization :
```bash
--model_type "resnet_gn"
```

## Test

For fedavg/fedbn :
```bash
python main.py --communications 200 --local_epochs 1 \
    --model_type "resnet" --learning_rate 0.001 --algorithm "fedavg" --seed 0 \
    --load_dir "/home/edlab/sjyang/temp/data_storage/federated_ecg_manifest" --test
```

For fedprox/fedpxn :
```bash
python main.py --communications 200 --local_epochs 1 \
    --model_type "resnet" --learning_rate 0.001 --algorithm "fedprox" --mu 0.01 --seed 0 \
    --load_dir "/home/edlab/sjyang/temp/data_storage/federated_ecg_manifest" --test
 ```

For fedadam :
```bash
python main.py --communications 200 --local_epochs 1 \
    --model_type "resnet" --learning_rate 0.001 --algorithm "fedadam" --seed 0 \
    --server_learning_rate 0.01 --tau 0.01 \
    --load_dir "/home/edlab/sjyang/temp/data_storage/federated_ecg_manifest" --test
```
