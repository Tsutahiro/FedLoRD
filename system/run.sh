#!/bin/bash
# FedLoRD example commands. Run from system/ after generating the data partitions (see ../dataset/README.md).
# Outputs: ../results/<dataset>_FedLoRD_<goal>_<trial>.h5 and ../results/<dataset>_FedLoRD_<goal>_localtime.json
cd "$(dirname "$0")"

# ------------------------------------------------------------------------------------------------
# Paper protocol: every client performs exactly K=100 local mini-batch updates per round
# (T_Theta on the shared part, then T_PQ on the personal factors), batch 100, lr 0.1,
# 100 rounds, 20 clients, 3 trials (seeds 0,1,2).
# ------------------------------------------------------------------------------------------------

# CIFAR-10, Dir(0.1):  lambda = 0.1, beta = 0.1, T_Theta : T_PQ = 3 : 2  (-kth 0.6 -> 60 + 40 steps)
python main.py -data Cifar10 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.6 -lr 0.1 -gr 100 \
       -nlam 0.1 -rkr 0.1 -sd 0 -t 3 -did 0 --require_cuda -go c10_dir01

# CIFAR-10, Dir(0.5):  lambda = 0.1, beta = 0.7, T_Theta : T_PQ = 2 : 3  (-kth 0.4 -> 40 + 60 steps)
python main.py -data Cifar10_a05 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.4 -lr 0.1 -gr 100 \
       -nlam 0.1 -rkr 0.7 -sd 0 -t 3 -did 0 --require_cuda -go c10_dir05

# CIFAR-100 / Tiny-ImageNet with ResNet-18 (GroupNorm): same protocol; set -data/-m/-ncl and tune -nlam/-rkr/-kth
# python main.py -data Cifar100     -m ResNet18 -ncl 100 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.6 -lr 0.1 -gr 100 -nlam 0.1 -rkr 0.1 -sd 0 -t 3 -did 0 --require_cuda -go c100_dir01
# python main.py -data TinyImagenet -m ResNet18 -ncl 200 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.6 -lr 0.1 -gr 100 -nlam 0.1 -rkr 0.1 -sd 0 -t 3 -did 0 --require_cuda -go ti_dir01

# ------------------------------------------------------------------------------------------------
# Legacy PFLlib protocol: local epochs instead of a fixed number of updates
# (5 local epochs = 4 on the shared part + 1 on the personal factors, batch 128).
# ------------------------------------------------------------------------------------------------
python main.py -data Cifar10 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 128 -ls 5 -EO 4 -EPQ 1 -lr 0.1 -gr 100 \
       -nlam 0.1 -rkr 0.1 -sd 0 -did 0 --require_cuda -go c10_legacy

# Ablation: no variable separation (full Delta_i with proximal SGD / singular-value thresholding)
# python main.py -data Cifar10 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.6 -lr 0.1 -gr 100 -nlam 0.1 -pqu svd -sd 0 -did 0 --require_cuda -go c10_svd
