# Memory-Efficient RkNN Retrieval by Nonlinear k-Distance Approximation

[![arXiv](https://img.shields.io/badge/arXiv-2011.01773-b31b1b)](https://arxiv.org/abs/2011.01773)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-2d618c?logo=python)](https://docs.python.org/3.8/)
[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the paper

```
Memory-Efficient RkNN Retrieval by Nonlinear k-Distance Approximation
Sandra Obermeier, Max Berrendorf, and Peer Kröger
```


# Installation
Create virtualenv
```shell
python3.8 -m venv venv 
source venv/bin/activate
```

Install dependencies
```shell
pip install -U pip setuptools
pip install -U -r requirements.txt
```

## MLflow server
To start the MLFLow server call
```shell script
(venv) mlflow server --tracking_uri=${TRACKING_URI}
```
with appropriate `TRACKING_URI`.

# Experiments
To reproduce our experiments, run
```shell script
(venv) PYTHONPATH=./src python3 executables/main.py --dataset=${DATASET} --model=${MODEL} -- tracking_uri=${TRACKING_URI}
```
The runs will be logged to the running MLFlow instance.

# Evaluation
To evaluate the runs and produce the figures from our paper, use the notebooks provided in [`notebooks`](./notebooks)
