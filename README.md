# FedLoRD: Personalized Federated Learning based on Low-Rank Structures of Model Parameter Differences

This repository contains the official implementation of **FedLoRD**, the personalized
federated learning (PFL) method proposed in

> Tsutahiro Fukuhara, Junya Hara, Hiroshi Higashi, and Yuichi Tanaka,
> "FedLoRD: Personalized Federated Learning based on Low-Rank Structures of Model Parameter Differences",
> arXiv:2610.17621, 2026. [[arXiv]](https://arxiv.org/abs/2610.17621)

The code is built on top of [PFLlib](https://github.com/TsingZ0/PFLlib).

FedLoRD starts from the observation that the difference between a client's personalized
model and the global model is approximately low-rank at every layer. Local training therefore
regularizes the layer-wise parameter differences with a nuclear norm, and the problem is
rewritten through the factorized representation of the nuclear norm so that it can be solved
by plain stochastic gradient descent, without any SVD.

FedLoRD gives every client $i$ a personalized model
$X_i = \Theta + P_i Q_i$: a **shared backbone** $\Theta$ learned by all clients plus a
**low-rank personal correction** $\Delta_i = P_i Q_i$ that never leaves the client.
Low rank is encouraged with a nuclear-norm penalty $\lambda\,\lVert X_i - \Theta\rVert_*$,
which is handled through its variational form
$\lVert \Delta\rVert_* = \min_{PQ=\Delta} \tfrac12(\lVert P\rVert_F^2 + \lVert Q\rVert_F^2)$:
the regularizer becomes a plain ridge term on the factors, so no SVD and no proximal
operator are needed during training. Each round, a client receives $\Theta^t$ and runs a
Ditto-style bilevel local update: (i) $T_\Theta$ SGD steps on the shared part starting from
$\Theta^t$, which is what it uploads, and (ii) $T_{PQ}$ gradient steps on $(P_i, Q_i)$ at the
fixed reference $\Theta^t$,
$P_i \leftarrow P_i - \eta\,(g\,Q_i^\top + \lambda P_i)$,
$Q_i \leftarrow Q_i - \eta\,(P_i^\top g + \lambda Q_i)$ with
$g = \nabla f_i(\Theta^t + P_i Q_i)$. The server averages only the shared part (weighted by
client data size over the participating clients). Clients never upload $P_i$ or $Q_i$, so
the communication cost equals FedAvg. Every 2-D weight (convolution kernels are reshaped to
matrices, linear layers as they are) gets a factor pair $(P_i, Q_i)$ whose width is a
fraction $\beta$ of $\min(m, n)$; 1-D parameters (biases, GroupNorm affine parameters) get a
full personal vector with the same ridge.

<p align="center">
  <!-- TODO: add the method overview figure (e.g. docs/fedlord.png) -->
  <img src="docs/fedlord.png" width="640" alt="FedLoRD overview (figure placeholder)">
</p>

## Repository structure

```
FedLoRD/
├── dataset/                      data generation (PFLlib generators)
│   ├── generate_Cifar10.py
│   ├── generate_Cifar100.py
│   ├── generate_TinyImagenet.py
│   ├── utils/dataset_utils.py    Dirichlet / pathological partitioning, train-test split, saving
│   └── README.md
├── system/
│   ├── main.py                   entry point: arguments, model builder, trials, result averaging
│   ├── run.sh                    example commands
│   ├── flcore/
│   │   ├── clients/clientbase.py     base client (data loading, evaluation)
│   │   ├── clients/clientfedlord.py  FedLoRD client: bilevel local update, personal factors (P_i, Q_i)
│   │   ├── servers/serverbase.py     base server (client management, evaluation, result saving)
│   │   ├── servers/serverfedlord.py  FedLoRD server: broadcast Theta, aggregate the shared part only
│   │   └── trainmodel/models.py      FedAvgCNN, ResNet-18 (GroupNorm), fastText, HARCNN
│   └── utils/
│       ├── data_utils.py         reading the client partitions
│       └── result_utils.py       mean / std of the best accuracy over trials
├── requirements.txt
├── LICENSE                       Apache-2.0 (this code is built on PFLlib, Apache-2.0)
└── README.md
```

## Environment

Python >= 3.8 with PyTorch >= 2.0 (tested with Python 3.11, PyTorch 2.0.1, CUDA 11.7).

```bash
conda create -n fedlord python=3.11 -y
conda activate fedlord
pip install torch torchvision        # pick the build matching your CUDA version
pip install -r requirements.txt      # numpy<2, h5py, scikit-learn, ujson
```

## Data

Generate the non-IID partitions first (details and all settings in
[`dataset/README.md`](dataset/README.md)):

```bash
cd dataset
python generate_Cifar10.py noniid - dir                   # 20 clients, Dir(0.1)  -> dataset/Cifar10
python generate_Cifar10.py noniid - dir 0.5 Cifar10_a05   # 20 clients, Dir(0.5)  -> dataset/Cifar10_a05
python generate_Cifar100.py noniid - dir
python generate_TinyImagenet.py noniid - dir
```

Each client's data is split 75 % / 25 % into a local train and a local test set; all
reported accuracies are on the clients' local test sets.

## Training

All commands are run from `system/` (see `run.sh`). Results are written to `../results/`.

**Equal-update protocol (used in the paper).** Every client performs exactly `-li K`
local mini-batch updates per round, split into `T_Θ = round(kth·K)` steps on the shared
part followed by `T_PQ = K − T_Θ` steps on the personal factors.

```bash
cd system
# CIFAR-10, Dir(0.1): lambda = 0.1, beta = 0.1, T_Theta : T_PQ = 3 : 2
python main.py -data Cifar10 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.6 \
       -lr 0.1 -gr 100 -nlam 0.1 -rkr 0.1 -sd 0 -t 3 -did 0 -go c10_dir01

# CIFAR-10, Dir(0.5): lambda = 0.1, beta = 0.7, T_Theta : T_PQ = 2 : 3
python main.py -data Cifar10_a05 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.4 \
       -lr 0.1 -gr 100 -nlam 0.1 -rkr 0.7 -sd 0 -t 3 -did 0 -go c10_dir05
```

**Legacy epoch protocol (PFLlib style).** Local work is counted in epochs: `-EO` epochs on
the shared part and `-EPQ` epochs on the personal factors (`-ls` is used as the fallback for
either of them when it is set to a value <= 0).

```bash
python main.py -data Cifar10 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 128 -ls 5 -EO 4 -EPQ 1 \
       -lr 0.1 -gr 100 -nlam 0.1 -rkr 0.1 -sd 0 -did 0 -go c10_legacy
```

`-t 3` runs three independent trials with seeds `-sd`, `-sd+1`, `-sd+2` and prints the
mean and standard deviation of the best test accuracy at the end. Pass `--require_cuda`
to abort instead of silently running on the CPU when no GPU is visible.

### Arguments

FedLoRD-specific:

| argument | meaning | default |
|---|---|---|
| `-nlam` | nuclear-norm coefficient $\lambda$ (ridge coefficient on $P_i$, $Q_i$ and on the 1-D personal vectors) | 0.002 |
| `-rkr` | factor width $\beta$: per layer $r = \max(1, \mathrm{round}(\beta \cdot \min(m,n)))$; `-1` = off | -1 |
| `-rk` | absolute rank upper bound $r$ (used only when `-rkr` is off); `-1` = $\min(m,n)$ | -1 |
| `-li` | $K$: local mini-batch updates per round (equal-update protocol); `-1` = legacy epochs | -1 |
| `-kth` | fraction of $K$ spent on the shared part: $T_\Theta = \mathrm{round}(\texttt{kth}\cdot K)$, $T_{PQ} = K - T_\Theta$ | 0.8 |
| `-EO` / `-EPQ` | legacy protocol: epochs on the shared part / on the factors | 4 / 1 |
| `-etapq` | step size $\eta$ of the $(P,Q)$ update; `-1` = the current (decayed) local learning rate | -1 |
| `-pqu` | factor update: `jacobi` (simultaneous $P$, $Q$ update from one gradient), `gs` (Gauss-Seidel: update $P$, recompute the gradient, update $Q$), `svd` (ablation without the variational form: a full $\Delta_i$ updated by proximal SGD with singular-value thresholding) | jacobi |
| `-nlame` | optional separate $\lambda$ for embedding layers (parameter names containing `embedding`); `-1` = off | -1 |

General (PFLlib): `-data` dataset folder, `-m` model (`CNN`, `ResNet18`, `fastText`,
`HARCNN`), `-ncl` number of classes, `-nc` number of clients (must match the partition),
`-jr` participation ratio per round, `-lbs` batch size, `-lr` local learning rate,
`-ld`/`-ldg` exponential learning-rate decay (on/off, gamma), `-wd` weight decay of the
shared-part SGD, `-gr` rounds, `-eg` evaluation gap, `-t` trials, `-sd` seed, `-go` run tag,
`-rdir` output directory, `-did` GPU id, `-fd`/`-vs` fastText embedding dimension and
vocabulary size. `python main.py -h` lists everything.

### Output files

For a run with tag `-go <goal>` on dataset `<data>` the following files are written to
`-rdir` (default `../results/`):

* `<data>_FedLoRD_<goal>_<trial>.h5` -- per-round `rs_test_acc`, `rs_test_auc`
  (sample-weighted averages over all clients' personalized models on their local test
  sets) and `rs_train_loss`.
* `<data>_FedLoRD_<goal>_localtime.json` -- per client, per round wall-clock time of the
  local update as `[round, t_lower, t_upper, t_total]` seconds, where `t_lower` is the
  shared-part update and `t_upper` the $(P,Q)$ update.
* `system/models/<data>/FedLoRD_server.pt` -- the final shared model $\Theta$.

The console prints, per round, the averaged train loss, test accuracy and AUC and their
standard deviations across clients, and at the end the best accuracy of the run.

## Results

Best personalized test accuracy (%) over 100 rounds, mean ± std over 3 trials, 20 clients,
FedAvgCNN, equal-update protocol with $K = 100$, batch size 100, learning rate 0.1:

| dataset | partition | $\lambda$ | $\beta$ | $T_\Theta : T_{PQ}$ | accuracy |
|---|---|---|---|---|---|
| CIFAR-10 | Dir(0.1) | 0.1 | 0.1 | 3 : 2 (`-kth 0.6`) | 91.22 ± 0.19 |
| CIFAR-10 | Dir(0.5) | 0.1 | 0.7 | 2 : 3 (`-kth 0.4`) | 79.29 ± 0.13 |

<!-- TODO: add the remaining datasets of the paper -->

## Citation

If you use this code, please cite the FedLoRD paper:

```bibtex
@article{fukuhara2026fedlord,
  title   = {{FedLoRD}: Personalized Federated Learning based on Low-Rank Structures of Model Parameter Differences},
  author  = {Fukuhara, Tsutahiro and Hara, Junya and Higashi, Hiroshi and Tanaka, Yuichi},
  journal = {arXiv preprint arXiv:2610.17621},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.17621}
}
```

## Acknowledgement

This implementation is built on [PFLlib](https://github.com/TsingZ0/PFLlib) (Apache-2.0):
the base client and server classes, the dataset generators, the model definitions and the
evaluation utilities are taken from it. Please also cite PFLlib:

```bibtex
@article{zhang2025pfllib,
  title   = {{PFLlib}: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark},
  author  = {Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal = {Journal of Machine Learning Research},
  volume  = {26},
  number  = {50},
  pages   = {1--10},
  year    = {2025}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
