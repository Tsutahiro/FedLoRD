# FedLoRD

Official implementation of **FedLoRD: Personalized Federated Learning based on Low-Rank Structures of Model Parameter Differences** ([arXiv:2610.17621](https://arxiv.org/abs/2610.17621)).

## Setup

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Data

```bash
cd dataset
python generate_Cifar10.py noniid - dir
```

See [dataset/README.md](dataset/README.md) for other datasets.

## Run

```bash
cd system
python main.py -data Cifar10 -m CNN -ncl 10 -nc 20 -algo FedLoRD -lbs 100 -li 100 -kth 0.6 \
       -lr 0.1 -gr 100 -nlam 0.1 -rkr 0.1 -sd 0 -t 3 -did 0 -go c10_dir01
```

More examples: [system/run.sh](system/run.sh)

## Citation

```bibtex
@article{fukuhara2026fedlord,
  title   = {{FedLoRD}: Personalized Federated Learning based on Low-Rank Structures of Model Parameter Differences},
  author  = {Fukuhara, Tsutahiro and Hara, Junya and Higashi, Hiroshi and Tanaka, Yuichi},
  journal = {arXiv preprint arXiv:2610.17621},
  year    = {2026}
}
```

## Acknowledgement

Built on [PFLlib](https://github.com/TsingZ0/PFLlib).

## License

Apache License 2.0
