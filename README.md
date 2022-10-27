# DR-VIDAL - Doubly Robust Variational Information(Theoretic) Adversarial Learning for the Estimation of Treatment Effects and Counterfactuals
# **** Update ****:
Paper accepted in American Medical Informatics Association Annual Symposium (AMIA)' 22 (Oral). Information related to citation will be posted soon.

## Keywords:
causal AI, biomedical informatics, generative adversarial networks, variational inference, information theory, doubly robust

## Supplementary Material for the paper
The supplementary material with proofs and additional results will be found at: [here](https://github.com/Shantanu48114860/DR-VIDAL/blob/main/DR_VIDAL_AMIA-Supp.pdf).

## Presentation slides
The presentation slides will be found at: 
1. [PPT](https://github.com/Shantanu48114860/DR-VIDAL-AMIA-22/blob/main/AMIA2022-powerpoint.pptx)
2. [PDF](https://github.com/Shantanu48114860/DR-VIDAL-AMIA-22/blob/main/AMIA2022-powerpoint.pdf)

## Packages
All the packages are inluded in **environment.yml** file

## Overview
<img src="https://github.com/Shantanu48114860/DR-VIDAL-AMIA-22/blob/main/DR_VIDAL.png"/>

## Requirements and versions
- pytorch - 1.3.1
- numpy - 1.17.2 
- pandas - 0.25.1 
- scikit - 0.21.3 
- matplotlib - 3.1.1 
- python -  3.8

## Dependencies
[python 3.8](https://www.python.org/downloads/release/)

[pytorch 1.3.1](https://pytorch.org/get-started/previous-versions/)

## How to run
First go the folder **DR_Info_CFR** by the command **cd DR_Info_CFR** and then do the following for each of the 3 datasets:

- IHDP: 

Command to reproduce the experiments mentioned in the paper for IHDP dataset:

cd IHDP 

python3 main_IHDP.py

- Jobs: 

Command to reproduce the experiments mentioned in the paper for Jobs dataset:

  cd Jobs 
  
  python3 main_Jobs.py

- Twins: 

Command to reproduce the experiments mentioned in the paper for Twins dataset:

  cd Twins 
  
  python3 main_Twins.py


## Hyperparameters
 - IHDP:
 IHDP/Constants.py
 
 - Jobs:
 Jobs/Constants.py
 
 - Twins: 
 Twins/Constants.py
 
## Results
<img src="https://github.com/Shantanu48114860/DR-VIDAL-AMIA-22/blob/main/Results_1.png"/>

<img src="https://github.com/Shantanu48114860/DR-VIDAL-AMIA-22/blob/main/Results_2.png"/>

## License & copyright

Licensed under the [MIT License](LICENSE)

Copyright (c) [DISL](https://epidemiology.phhp.ufl.edu/research/disl), 2021
