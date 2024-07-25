# GNN-IDS

Code for the paper "GNN-IDS: Graph Neural Networks for Intrusion Detection Systems" that was accepted at the International Conference on Availability, Reliability and Security ([ARES 2024](https://www.ares-conference.eu/)).

## Requirements
* Python >= 3.8.10
* PyTorch >= 2.0.1+cu117
* PyTorch Geometric >= 2.3.1

## Install required packages
```bash
bash install.sh
```
## Use case network
* The use case network studied in our work can be found in files under folder `mulval_attack_graph`, in which `.dot` file contains the texual representation of the network and `.pdf` file contains the visual representation of the network.
* Refer to [MulVAL](https://github.com/risksense/mulval) for more information about how to generate the attack graph.


## Datasets

*  Dataset 1 is a synthetic dataset. It is stored in the `datasets/synt` folder.
* Dataset 2 is generated based on the public [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html). It is stored in the `datasets/public` folder.

*Note: the datasets should be loaded in notebooks to reproduce the experimental results.*
## How to run the code
We provide two notebooks to run the code:
* `gnn_ids_dataset1.ipynb` for Dataset 1
* `gnn_ids_dataset2.ipynb` for Dataset 2

## Citation
If you find this code or our work useful in your research, please consider citing our paper:
Bibtex - TODO
```
```
