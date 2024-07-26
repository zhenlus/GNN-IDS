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
```
@inproceedings{10.1145/3664476.3664515,
author = {Sun, Zhenlu and Teixeira, Andr\'{e} M.H. and Toor, Salman},
title = {GNN-IDS: Graph Neural Network based Intrusion Detection System},
year = {2024},
isbn = {9798400717185},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664476.3664515},
doi = {10.1145/3664476.3664515},
abstract = {Intrusion detection systems (IDSs) are widely used to identify anomalies in computer networks and raise alarms on intrusive behaviors. ML-based IDSs generally take network traces or host logs as input to extract patterns from individual samples, whereas the inter-dependencies of network are often not captured and learned, which may result in large amounts of uncertain predictions, false positives, and false negatives. To tackle the challenges in intrusion detection, we propose a graph neural network-based intrusion detection system (GNN-IDS), which is data-driven and machine learning-empowered. In our proposed GNN-IDS, the attack graph and real-time measurements that represent static and dynamic attributes of computer networks, respectively, are incorporated and associated to represent complex computer networks. Graph neural networks are employed as the inference engine for intrusion detection. By learning network connectivity, graph neural networks can quantify the importance of neighboring nodes and node features to make more reliable predictions. Furthermore, by incorporating an attack graph, GNN-IDS could not only detect anomalies but also identify the malicious actions causing the anomalies. The experimental results on a use case network with two synthetic datasets (one generated from public IDS data) show that the proposed GNN-IDS achieves good performance. The results are analyzed from the aspects of uncertainty, explainability, and robustness.},
booktitle = {Proceedings of the 19th International Conference on Availability, Reliability and Security},
articleno = {14},
numpages = {12},
keywords = {Explainability, Graph Neural Network, Intrusion Detection System, Robustness, Uncertainty},
location = {Vienna, Austria},
series = {ARES '24}
}
```
