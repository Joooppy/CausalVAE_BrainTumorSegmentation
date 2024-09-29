![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg)

# CVAE: Causal VAE for MRI Brain Tumor Segmentation (Work in Progress)

This project combines a variational autoencoder regularized brain tumor segmentation network with a causal mask to discover causal structure and inference to generate synthetic data. The dataset used for training is the Adult Glioma dataset from the BraTS 2ß18 Challenge. (https://www.med.upenn.edu/sbia/brats2018/data.html)


The base segmentation network used is the BraTS challenge winner from 2018 Myronenko A.: (https://arxiv.org/pdf/1810.11654.pdf) using an adapted version of the implementation by: 
C. Lyu, H. Shu. (2021) A Two-Stage Cascade Model with Variational Autoencoders and Attention Gates for MRI Brain Tumor Segmentation. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2020), LNCS 12658, pp. 435-447. (https://arxiv.org/abs/2011.02881) <br>
Github: (https://github.com/shu-hai/two-stage-VAE-Attention-gate-BraTS2020)

 
The variational autoencoder part is changed into a causal graph mask adapted to the task at hand from:
Yang M. et al., (2023) CausalVAE: Structured Causal Disentanglement in Variational Autoencoder (https://arxiv.org/pdf/2004.08697v7) <br>
Github: (https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE)

Due to technical and time limitations the project is currently on hold.

<br /><br />




# Data Preparation

To prepare data from the start:
1. Normalization:
-puts multiple modalities(.nii.gz files) and the segmentation label into a .npy file for each patient
-normalizes the MRI images to have mean 0 and std 1

```bash
cd path_to_code
python normalization.py -y 2018
```
Use `-y 2020` for 2020 data

2. Train Test Partition:
Use the script for split. the partition result is store in train_list.txt and valid_list.txt.
```bash
python train_test_split.py
```

The additionally filter the data to have a to at least have proportion of 0.005 compared to the brainmask of each individual segmentation label, so to have more meaningful data, use the data_prep notebooks and adjust the path accordingly.


# Training
- For training the model, use:
```bash
cd CausalVAE
python main.py -e num_epochs -g num_gpus -p load_old_model
``` 
The model will save the best model, log state dicts and saves the input and reconstruction image of the current epoch.


# To Do

The next steps will be:
- full training run and evaluate resulting causal DAG
- test learning rates between 0.001 and 0.0001 to optimize DAG learning
- change between label distribution and label proportion as a weak supervision for the DAG and evaluate the forced bidirectional causal effect
- adjust model inference mask for the task to enable controlled inference
- inference to control synthetic data generation and analyse latent space
