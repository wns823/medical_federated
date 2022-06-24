data_path=$1

unzip ${data_path}/ISIC_2019_Training_Input.zip
mv ${data_path}/ISIC_2019_Training_Input ${data_path}/ISIC_2019
mv ${data_path}/ISIC_2019_Training_GroundTruth.csv ${data_path}/ISIC_2019/