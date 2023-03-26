# Solitude

This repository is the official implementation of the paper [Towards Private Learning on Decentralized Graphs with Local Differential Privacy](). 

**Abstract**

> 	Many real-world networks are inherently decentralized. For example, in social networks, each user maintains a local view of a social graph, such as a list of friends and her profile. It is typical to collect these local views of social graphs and conduct graph learning tasks. However, learning over graphs can raise privacy concerns as these local views often contain sensitive information. In this paper, we seek to ensure private graph learning on a decentralized network graph. Towards this objective, we propose Solitude, a new privacy-preserving learning framework based on graph neural networks (GNNs), with formal privacy guarantees based on edge local differential privacy. The crux of Solitude is a set of new delicate mechanisms that can calibrate the introduced noise in the decentralized graph collected from the users. The principle behind the calibration is the intrinsic properties shared by many real-world graphs, such as sparsity. Unlike existing work on locally private GNNs, our new framework can simultaneously protect node feature privacy and edge privacy, and can seamlessly incorporate with any GNN with privacy-utility guarantees. Extensive experiments on benchmarking datasets show that Solitude can retain the generalization capability of the learned GNN while preserving the users’ data privacy under given privacy budgets.

## Requirements

This code is implemented in Python 3.9, and relies on the following packages:  
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.8.1
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.7.0
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.2.4
- [Numpy](https://numpy.org/install/) >= 1.20.2
- [Seaborn](https://seaborn.pydata.org/) >= 0.11.1  
- [wandb](https://wandb.ai/site)

## Usage

### Training individual models
If you want to individually train and evaluate the models on any of the datasets mentioned in the paper, run the following command:  
```
python main.py -d cora -ee 8 -ex 1 -kx 4 -ky 2 --model sage --learning_rate 0.01 --weight_decay 0.001 --dropout 0.5 -s 1234 --max-epochs 500 -o ./out/cora/ee8_ex1

python main.py -d citeseer -ee 8 -ex 1 -kx 4 -ky 2 --model sage --learning_rate 0.01 --weight_decay 0.001 --dropout 0.5 -s 12345 --max-epochs 200 --orphic True -o ./out/citeseer/ee8_ex1

python main.py -d lastfm -ee 8 -ex 1 -kx 4 --model sage --learning_rate 0.01 --weight_decay 0.001 --dropout 0.5 -s 12345 --max-epochs 500 -o ./out/lastfm/ee8_ex1

dataset arguments:
  -d              <string>       name of the dataset (choices: cora, pubmed, facebook, lastfm) (default: cora)
  --data-dir      <path>         directory to store the dataset (default: ./datasets)
  --data-range    <float pair>   min and max feature value (default: (0, 1))
  --val-ratio     <float>        fraction of nodes used for validation (default: 0.25)
  --test-ratio    <float>        fraction of nodes used for test (default: 0.25)

data transformation arguments:
  -ex             <float>        privacy budget for feature perturbation (default: inf)
  -ee             <float>        privacy budget for edge perturbation (default: inf)

model arguments:
  --model         <string>       backbone GNN model (choices: gcn, sage, gat) (default: sage)
  --hidden-dim    <integer>      dimension of the hidden layers (default: 16)
  --dropout       <float>        dropout rate (between zero and one) (default: 0.0)
  -kx             <integer>      KProp step parameter for features (default: 0)
  -ky             <integer>      KProp step parameter for labels (default: 0)
  --forward       <boolean>      applies forward loss correction (default: True)

trainer arguments:
  --optimizer     <string>       optimization algorithm (choices: sgd, adam) (default: adam)
  --max-epochs    <integer>      maximum number of training epochs (default: 500)
  --learning-rate <float>        learning rate (default: 0.01)
  --weight-decay  <float>        weight decay (L2 penalty) (default: 0.0)
  --patience      <integer>      early-stopping patience window size (default: 0)
  --device        <string>       desired device for training (choices: cuda, cpu) (default: cuda)
  --orphic        <string>       use estimator to clean graph (choices: True, False) (default: False)
  --inner_epoch   <integer>      the time of using estimator in a single epoch (default: 2)
  --checkpoint_dir <string>      the path to save checkpoint

experiment arguments:
  -s              <integer>      initial random seed (default: None)
  -r              <integer>      number of times the experiment is repeated (default: 1)
  -o              <path>         directory to store the results (default: ./output)
  --log           <boolean>      enable wandb logging (default: False)
  --log-mode      <string>       wandb logging mode (choices: individual,collective) (default: individual)
  --project-name  <string>       wandb project name (default: Solitude)
  --retrain       <boolean>      whether to retrain the model (default: False)
```

The test result for each run will be saved as a csv file in the directory specified by  
``-o`` option (default: ./output).

## Citation

If you find this code useful, please cite the following paper: 
```bibtex 
@article{lin2022towards,
  title={Towards private learning on decentralized graphs with local differential privacy},
  author={Lin, Wanyu and Li, Baochun and Wang, Cong},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={2936--2946},
  year={2022},
  publisher={IEEE}
}
```
