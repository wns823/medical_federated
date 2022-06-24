wget -O WFDB_CPSC2018.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/
wget -O WFDB_CPSC2018_2.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018_2.tar.gz/
wget -O WFDB_StPetersburg.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTB.tar.gz/
wget -O WFDB_PTBXL.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTBXL.tar.gz/
wget -O WFDB_Ga.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/
wget -O WFDB_ChapmanShaoxing.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ChapmanShaoxing.tar.gz/
wget -O WFDB_Ningbo.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ningbo.tar.gz/

tar -xvf WFDB_ChapmanShaoxing.tar.gz
tar -xvf WFDB_CPSC2018.tar.gz
tar -xvf WFDB_CPSC2018_2.tar.gz
tar -xvf WFDB_Ga.tar.gz
tar -xvf WFDB_Ningbo.tar.gz
tar -xvf WFDB_PTB.tar.gz
tar -xvf WFDB_PTBXL.tar.gz

rm WFDB_ChapmanShaoxing.tar.gz
rm WFDB_CPSC2018.tar.gz
rm WFDB_CPSC2018_2.tar.gz
rm WFDB_Ga.tar.gz
rm WFDB_Ningbo.tar.gz
rm WFDB_PTB.tar.gz
rm WFDB_PTBXL.tar.gz
