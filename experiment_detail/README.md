# The details of experiment

We show the implementation details of all experiments in the page. 

## Batch size

We use the same batch size for all FL methods in each dataset.

|   Dataset   | batch size |
| ----------- | ---------- |
| eICU        |    256     |
| skin cancer |    128     | 
| ECG         |     64     |

## Hyperparameters for FL methods

We utilized the grid search map used in the existing papers(FedProx, FedOpt, FedDyn).
- learning rate ($\eta$)
    - := [0.1, 0.01, 0.001, 0.0001] (FedAvg, FedProx, FedBn, FedPxN, FedDyn) 
    - := [0.1, 0.03, 0.01, 0.003, 0.001, 0.0001] (FedAdam, FedAdagrad, FedYogi)
- mu ($\mu$) := [1.0, 0.1, 0.01, 0.001, 0.0001]
- feddyn alpha ($\alpha$) := [0.0001, 0.001, 0.01, 0.1]
- server learning rate ($\eta_{g}$) := [0.1, 0.03, 0.01, 0.003, 0.001, 0.0001]
- tau ($\gamma$) := [0.0001, 0.001, 0.01, 0.1]

For the grid search, we used 10% of total data in skin cancer & ECG and the data of 5 largest clients in eICU. You can check the used hyperparameters in below links.
- Mortality prediction ([mort_24h](eICU_mort_24h.csv), [mort_48h](eICU_mort_48h.csv))
- Length of Stay prediction ([LOS](eICU_LOS.csv))
- Imminent discharge prediction ([disch_24h](eICU_disch_24h.csv), [disch_48h](eICU_disch_48h.csv))
- Final Acuity prediction ([Acuity](eICU_Final_Acuity.csv))
- Skin cancer classification ([skin_cancer_image](skin_cancer_images.csv))
- Cardiac arrhythmia classification ([ECG](ECG.csv))

