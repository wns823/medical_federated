data_path=$1

mkdir ${data_path}/federated_ecg_manifest

python fairseq_signals/data/ecg/preprocess/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "ChapmanShaoxing" \
    --combine_subsets "ChapmanShaoxing" \
    --dest "${data_path}/federated_ecg_manifest/ChapmanShaoxing" \
    --valid-percent 0.1

python fairseq_signals/data/ecg/preprocess/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "CPSC2018, CPSC2018_2" \
    --combine_subsets "CPSC2018, CPSC2018_2" \
    --dest "${data_path}/federated_ecg_manifest/CPSC2018" \
    --valid-percent 0.1

python fairseq_signals/data/ecg/preprocess/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "Ga" \
    --combine_subsets "Ga" \
    --dest "${data_path}/federated_ecg_manifest/Ga" \
    --valid-percent 0.1

python fairseq_signals/data/ecg/preprocess/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "Ningbo" \
    --combine_subsets "Ningbo" \
    --dest "${data_path}/federated_ecg_manifest/Ningbo" \
    --valid-percent 0.1

python fairseq_signals/data/ecg/preprocess/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "PTBXL" \
    --combine_subsets "PTBXL" \
    --dest "${data_path}/federated_ecg_manifest/PTBXL" \
    --valid-percent 0.1
