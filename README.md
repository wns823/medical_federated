# Towards the Practical Utility of Federated Learning in the Medical Domain (CHIL 23)


**Seongjun Yang<sup>1</sup><sup>*</sup>, Hyeonji Hwang<sup>2</sup><sup>*</sup>, Daeyoung Kim<sup>3</sup>, Radhika Dua<sup>2</sup>, Jong-Yeup Kim<sup>4</sup>, Eunho Yang<sup>2</sup>, Edward Choi<sup>2</sup> ** | [Paper](https://arxiv.org/abs/2207.03075)

**<sup>1</sup>KRAFTON, <sup>2</sup>KAIST AI, <sup>3</sup>NCSOFT, <sup>4</sup>College of Medicine, Konyang University**

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
