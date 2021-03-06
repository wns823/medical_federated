# Towards the Practical Utility of Federated Learning in the Medical Domain

**Seongjun Yang<sup>1</sup>, Hyeonji Hwang<sup>1</sup>, Daeyoung Kim<sup>1</sup>, Radhika Dua<sup>1</sup>, Jong-Yeup Kim<sup>2</sup>, Eunho Yang<sup>1</sup>, Edward Choi<sup>1</sup>**

**<sup>1</sup>KAIST, <sup>2</sup>Department of Biomedical Informatics, College of Medicine, Konyang University**

This repository is official implementation of "Towards the Practical Utility of Federated Learning in the Medical Domain" ([link](https://arxiv.org/pdf/2207.03075.pdf)).

## Abstract
Federated learning (FL) is an active area of research. One of the most suitable areas for adopting FL is the medical domain, where patient privacy must be respected. Previous research, however, does not fully consider who will most likely use FL in the medical domain. It is not the hospitals who are eager to adopt FL, but the service providers such as IT companies who want to develop machine learning models with real patient records. Moreover, service
providers would prefer to focus on maximizing the performance of the models at the lowest cost possible. In this work, we propose empirical benchmarks of FL methods considering both performance and monetary cost with three real-world datasets: electronic health records, skin cancer images, and electrocardiogram datasets. We also propose Federated learning with Proximal regularization eXcept local Normalization (FedPxN), which, using a simple combination of FedProx and FedBN, outperforms all other FL algorithms while consuming only slightly more power than the most power efficient method.

## Federated learning with eICU database
We evaluate FL methods on the [eICU database](https://www.nature.com/articles/sdata2018178). The database is a multi-center intensive care unit (ICU)database with high granularity data for over 200,000 admissions to ICUs monitored by eICU Programs across the United States. We implement six clinical prediction tasks from the eICU database. 

You can see the details of FL with eICU [here](ehr_federated/README.md).



## Federated learning with skin cancer images
We evaluate FL methods on skin cancer image datasets originating from different sources. We implement skin cancer image classification. 

You can see the details of FL with skin cancer [here](skin_cancer_federated/README.md).

## Federated learning with Electrocardiogram (ECG)
We evaluate FL methods on the [PhysioNet 2021](https://moody-challenge.physionet.org/2021/) challenge dataset. The dataset contains ECG data originating from different sources. We implement the cardiac arrhythmia classification.

You can see the details of FL with ECG [here](ecg_federated/README.md).

## The details of Experiment
We have attached all details of the experiments in the [experiment_detail](experiment_detail/README.md).

## More experiment results
We attach the PRAUC results of FL methods with the eICU in the [experiment_results](experiment_results/README.md). 

# Contact
If you have any question or recommendation, please contact us via an issue or an e-mail.
* seongjunyang@kaist.ac.kr
* localh@kaist.ac.kr
* daeyoung.k@kaist.ac.kr
