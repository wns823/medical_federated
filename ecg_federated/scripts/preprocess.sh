data_path=$1

python fairseq_signals/data/ecg/preprocess/preprocess_physionet2021.py \
    "${data_path}" \
    --meta-dir "." \
    --dest "${data_path}/ecg_preprocessed_data" \
    --subset "WFDB_CPSC2018, WFDB_CPSC2018_2, WFDB_Ga, WFDB_PTB, WFDB_PTBXL, WFDB_ChapmanShaoxing, WFDB_Ningbo" \
    --workers 8