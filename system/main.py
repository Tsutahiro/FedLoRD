#!/usr/bin/env python
"""
FedLoRD -- entry point (PFLlib-style).

Builds the model, creates the FedLoRD server (which creates the clients), runs the
federated training for `-t` independent trials (seed = -sd + trial index) and finally
averages the best test accuracy over the trials.
"""
import argparse
import os
import time
import random
import warnings
import logging
import numpy as np
import torch

from flcore.servers.serverfedlord import FedLoRD
from flcore.trainmodel.models import FedAvgCNN, resnet18_gn, fastText, HARCNN
from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def build_model(args):
    """Model selection (same rules as PFLlib's main.py for the four supported models)."""
    model_str = args.model

    if model_str == "CNN":
        if "MNIST" in args.dataset:
            return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
        elif "Cifar10" in args.dataset:          # Cifar10 and Cifar100 (32x32 RGB)
            return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)
        elif "Omniglot" in args.dataset:
            return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856)
        else:
            return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)

    elif model_str == "ResNet18":
        # ResNet-18 + GroupNorm. CIFAR-adapted 3x3 stem for 32x32 inputs (Cifar10/100, MNIST);
        # the standard ImageNet stem otherwise (e.g. Tiny-ImageNet).
        small_input = ("Cifar" in args.dataset) or ("MNIST" in args.dataset)
        return resnet18_gn(num_classes=args.num_classes, small_input=small_input)

    elif model_str == "fastText":
        return fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)

    elif model_str == "HARCNN":
        if args.dataset == 'HAR':
            return HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                          pool_kernel_size=(1, 2))
        elif args.dataset == 'PAMAP2':
            return HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                          pool_kernel_size=(1, 2))
        raise NotImplementedError(f"HARCNN: unknown dataset {args.dataset} (expected HAR or PAMAP2)")

    raise NotImplementedError(f"unknown model {model_str}; choose CNN, ResNet18, fastText or HARCNN")


def run(args):
    time_list = []

    for i in range(args.prev, args.times):
        # Re-seed per trial so each trial i is an independent, reproducible draw. The data
        # partition on disk is fixed; this varies model init / batch order / client selection.
        seed_i = args.seed + i
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        random.seed(seed_i)
        torch.cuda.manual_seed_all(seed_i)
        print(f"\n============= Running time: {i}th  (seed={seed_i}) =============")
        print("Creating server and clients ...")
        start = time.time()

        args.model = build_model(args).to(args.device)
        print(args.model)

        if args.algorithm == "FedLoRD":
            server = FedLoRD(args, i)
        else:
            raise NotImplementedError(f"unknown algorithm {args.algorithm}; this repository provides FedLoRD only")

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # mean / std of the best test accuracy over the trials
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times,
                 result_dir=args.result_dir)

    print("All done!")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLoRD: low-rank personalized federated learning (PFLlib-style).")
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="Tag of this experiment (used in the output file names)")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0",
                        help="CUDA device id (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument('--require_cuda', action='store_true', default=False,
                        help="Abort instead of silently falling back to CPU when CUDA is unavailable")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10",
                        help="Folder name under ../dataset/ (e.g. Cifar10, Cifar10_a05, Cifar100, TinyImagenet)")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN", choices=["CNN", "ResNet18", "fastText", "HARCNN"])
    parser.add_argument('-algo', "--algorithm", type=str, default="FedLoRD", choices=["FedLoRD"])
    parser.add_argument('-lbs', "--batch_size", type=int, default=10, help="Local mini-batch size")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=True,
                        help="Exponential learning-rate decay per round (also used for the P,Q step unless -etapq>0)")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.998)
    parser.add_argument('-wd', "--weight_decay", type=float, default=1e-3,
                        help="SGD weight decay of the lower (shared-part) update")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, help="For auto_break")
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=False,
                        help="Stop early once the best accuracy has not improved for top_cnt rounds")
    # local budget: equal-update protocol (-li K > 0) or legacy local epochs (-li -1)
    parser.add_argument('-li', "--local_iters", type=int, default=-1,
                        help="Equal-update protocol: number K of local mini-batch updates per round, identical for "
                             "every client (the data loader is cycled). -1 = legacy behaviour: -ls local epochs.")
    parser.add_argument('-kth', "--k_theta_frac", type=float, default=0.8,
                        help="With -li>0: fraction of K spent on the lower (shared Theta) update, T_Theta = round(kth*K); "
                             "the remaining T_PQ = K - T_Theta steps update the personal factors (P,Q).")
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Legacy protocol: fallback local epochs per round, used for -EO / -EPQ when those are set to <= 0")
    parser.add_argument('-EO', "--E_theta", type=int, default=4,
                        help="Legacy protocol: lower (shared Theta) local epochs per round")
    parser.add_argument('-EPQ', "--E_pq", type=int, default=1,
                        help="Legacy protocol: upper (P,Q) local epochs per round")
    # federation
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False,
                        help="Random number of participating clients per round (between join_ratio*N and N)")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients (must match the generated partition)")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous running times")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times (trials)")
    parser.add_argument('-sd', "--seed", type=int, default=0,
                        help="Base random seed; trial i (0..times-1) uses seed+i")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-fs', "--few_shot", type=int, default=0, help="Few-shot training samples per class (0 = all)")
    parser.add_argument('-rdir', "--result_dir", type=str, default="../results/",
                        help="Directory for the .h5 / _localtime.json outputs")
    # model-specific
    parser.add_argument('-fd', "--feature_dim", type=int, default=512, help="fastText: embedding / hidden dimension")
    parser.add_argument('-vs', "--vocab_size", type=int, default=80,
                        help="fastText: vocabulary size (32000 for the PFLlib AG News partition)")
    # system heterogeneity (PFLlib)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending the global model")
    # FedLoRD
    parser.add_argument('-nlam', "--nuc_lambda", type=float, default=0.002,
                        help="FedLoRD: nuclear-norm coefficient lambda (= ridge coefficient on the factors P,Q)")
    parser.add_argument('-rkr', "--rank_ratio", type=float, default=-1.0,
                        help="FedLoRD: factor width beta; per-layer r = max(1, round(beta*min(m,n))). -1 = off (use -rk). "
                             "1.0 == full width")
    parser.add_argument('-rk', "--rank", type=int, default=-1,
                        help="FedLoRD: absolute rank upper bound r of P_i Q_i (-1 = min(m,n) per layer); ignored if -rkr>0")
    parser.add_argument('-etapq', "--eta_pq", type=float, default=-1.0,
                        help="FedLoRD: learning rate of the upper (P,Q) update (-1 = the current, decayed local lr)")
    parser.add_argument('-pqu', "--pq_update", type=str, default="jacobi", choices=["jacobi", "gs", "svd"],
                        help="FedLoRD: P,Q update mode: 'jacobi' (simultaneous), 'gs' (Gauss-Seidel), or 'svd' "
                             "(ablation without variable separation: full Delta_i updated by proximal SGD / SVT)")
    parser.add_argument('-nlame', "--nlam_embed", type=float, default=-1.0,
                        help="FedLoRD: optional per-layer override of lambda for embedding layers (parameter name "
                             "contains 'embedding'); -1 = use nuc_lambda everywhere")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        if args.require_cuda:
            raise SystemExit("CUDA required (--require_cuda) but torch.cuda.is_available() is False; refusing to run on CPU.")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    run(args)
