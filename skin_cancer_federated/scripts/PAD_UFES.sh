data_path=$1

unzip ${data_path}/zr7vgbcyr2-1.zip -d ${data_path}/
mkdir ${data_path}/PAD-UFES-20
mv ${data_path}/images ${data_path}/PAD-UFES-20/
mv ${data_path}/metadata.csv ${data_path}/PAD-UFES-20/
unzip ${data_path}/PAD-UFES-20/images/imgs_part_1.zip -d ${data_path}//PAD-UFES-20/
unzip ${data_path}/PAD-UFES-20/images/imgs_part_2.zip -d ${data_path}//PAD-UFES-20/
unzip ${data_path}/PAD-UFES-20/images/imgs_part_3.zip -d ${data_path}//PAD-UFES-20/
rm -rf ${data_path}/PAD-UFES-20/images
mv ${data_path}/PAD-UFES-20/imgs_part_1/* ${data_path}/PAD-UFES-20/
mv ${data_path}/PAD-UFES-20/imgs_part_2/* ${data_path}/PAD-UFES-20/
mv ${data_path}/PAD-UFES-20/imgs_part_3/* ${data_path}/PAD-UFES-20/
rm -rf ${data_path}/PAD-UFES-20/imgs_part_1
rm -rf ${data_path}/PAD-UFES-20/imgs_part_2
rm -rf ${data_path}/PAD-UFES-20/imgs_part_3
