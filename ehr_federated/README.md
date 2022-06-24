# Federated learning with eICU database

## Requirements

* [PyTorch](https://pytorch.org) version >= 1.8.0
* Python version >= 3.7
* For training new models, you'll also need an NIVDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

```bash
$ conda create -n ehr_federated python=3.7 (optional)
$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install wandb
$ pip install pandas==0.24.2
$ pip install scikit-learn==1.0.2
```

## Prepare dataset

- Make the folder to save datasets
```bash
$ mkdir [your specific path]/data_storage && cd [your specific path]/data_storage
```

1. Download eICU database from the [link](https://eicu-crd.mit.edu/)
```
/path/to/eICU
├─ data_storage
  └─ eicu-2.0
```

2. Download benchmark dataset from the [this repository](https://github.com/mmcdermott/comprehensive_MTL_EHR). This dataset folders should be located in [data_storage_path]/eicu-2.0/federated_preprocessed_data
```
/path/to/benchmark_dataset
├─ data_storage
  └─ eicu-2.0
    └─ federated_preprocessed_data
        └─ final_datasets_for_sharing (benchmark dataset)
```

3. Run the following script to pre-process and cache the dataset.
```bash
$ python ehr_federated/preprocess.py --data_path [data_storage_path]
```

## Train
For fedavg/fedbn :
```bash
python main.py --algorithm "fedavg" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 --model_type "transformer_ln" \
    --learning_rate 0.01 --data_path [data_storage_path]
```

For fedprox/fedpxn :
```bash
python main.py --algorithm "fedprox" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 --model_type "transformer_ln" \
    --learning_rate 0.01 -mu 0.01 --data_path [data_storage_path]
```

For fedadam/fedadagrad/fedyogi :
```bash
python main.py --algorithm "fedadam" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 \
    --model_type "transformer_ln" --server_learning_rate 0.01 \
    --learning_rate 0.01 --tau 0.01 --data_path [data_storage_path]
```

For feddyn :
```bash
python main.py --algorithm "feddyn" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 \
    --model_type "transformer_ln" --feddyn_alpha 0.01 --learning_rate 0.01 \
    --data_path [data_storage_path]
```

* For model type :
    * transformer with layer normalization --> transformer_ln
    * transformer with group normalization --> transformer_gn


### Tasks
```bash
--task ['mort_24h' or 'mort_48h' or 'LOS' or 'disch_24h' or 'disch_48h' or 'Final Acuity Outcome']
```

### 5 / 10 / 20 / 30 client
For 5 largest clients :
```bash
--hospital_id "73.264.420.243.458"
```

For 10 largest clients :
```bash
--hospital_id "73.264.420.243.458.443.338.252.208.122"
```

For 20 largest clients :
```bash
--hospital_id "73.264.420.243.458.443.338.252.208.122.167.199.281.176.449.188.416.283.417.394"
```

For 30 largest clients :
```bash
--hospital_id "73.264.420.243.458.443.338.252.208.122.167.199.281.176.449.188.416.283.417.394.411.197.110.248.300.148.365.413.183.400"
```

## Test

For fedavg/fedbn :
```bash
python main.py --algorithm "fedavg" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 --model_type "transformer_ln" \
    --learning_rate 0.01 --data_path [data_storage_path] --test
```

For fedprox/fedpxn :
```bash
python main.py --algorithm "fedprox" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 --model_type "transformer_ln" \
    --learning_rate 0.01 -mu 0.01 --data_path [data_storage_path] --test
```

For fedadam/fedadagrad/fedyogi :
```bash
python main.py --algorithm "fedadam" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 \
    --model_type "transformer_ln" --server_learning_rate 0.01 \
    --learning_rate 0.01 --tau 0.01 --data_path [data_storage_path] --test
```

For feddyn :
```bash
python main.py --algorithm "feddyn" --communications 100 --local_epochs 1 \
    --task "disch_24h" --hospital_id "73.264.420.243.458" --seed 1234 \
    --model_type "transformer_ln" --feddyn_alpha 0.01 --learning_rate 0.01 \
    --data_path [data_storage_path] --test
```