# Dataset generation

The scripts in this folder are the PFLlib generators. Each one downloads the raw data,
merges the official train and test sets, partitions the samples over the clients with a
Dirichlet label distribution, and splits every client's data 75 % / 25 % into train / test.
The output is read by `system/main.py` through the `-data <folder>` argument.

```bash
cd dataset

# CIFAR-10, 20 clients, Dir(alpha = 0.1)      -> Cifar10/
python generate_Cifar10.py noniid - dir

# CIFAR-10, 20 clients, Dir(alpha = 0.5)      -> Cifar10_a05/
python generate_Cifar10.py noniid - dir 0.5 Cifar10_a05

# CIFAR-100 (alpha = 0.1 -> Cifar100/, alpha = 0.5 -> Cifar100_a05/)
python generate_Cifar100.py noniid - dir
python generate_Cifar100.py noniid - dir 0.5 Cifar100_a05

# Tiny-ImageNet (downloads tiny-imagenet-200.zip with wget and unzips it first)
python generate_TinyImagenet.py noniid - dir
python generate_TinyImagenet.py noniid - dir 0.5 TinyImagenet_a05
```

Command-line form: `python generate_<X>.py <noniid|iid> <balance|-> <dir|pat|exdir|-> [alpha] [out_dir]`.
The first three arguments are the PFLlib ones (`noniid - dir` = unbalanced Dirichlet partition);
the optional `alpha` overrides the Dirichlet concentration (default `alpha = 0.1` in
`utils/dataset_utils.py`) and `out_dir` the output folder.

Settings fixed inside the scripts (edit them if you need something else):

| setting | value | where |
|---|---|---|
| number of clients | 20 | `num_clients` in each `generate_*.py` (pass the same value to `main.py -nc`) |
| Dirichlet alpha | 0.1 (or the 4th CLI argument) | `alpha` in `utils/dataset_utils.py` |
| train / test split per client | 0.75 / 0.25 | `train_ratio` in `utils/dataset_utils.py` |
| minimum samples per client | `min(batch_size/(1-train_ratio), N/num_clients/2)` with `batch_size = 10` | `utils/dataset_utils.py` |
| random seed | 1 (`random.seed(1)`, `np.random.seed(1)`) | each `generate_*.py` |
| image preprocessing | `ToTensor` + `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))` | each `generate_*.py` |

Output layout (`<out_dir>/config.json`, `<out_dir>/train/<client>.npz`, `<out_dir>/test/<client>.npz`):
`config.json` records the partition settings and the per-client label histogram; each `.npz`
holds a dict `{'x': images, 'y': labels}`. A generator refuses to overwrite a folder whose
`config.json` already matches the requested settings.

The generated folders (and the `rawdata/` downloads) are large and are ignored by `.gitignore`.
