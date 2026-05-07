# BiomedCoOp

BiomedCoOp is a prompt-learning repository for biomedical vision-language models built on top of BiomedCLIP, CLIP, Dassl, and OpenCLIP. This repo supports:

- centralized few-shot training and evaluation
- base-to-novel evaluation
- multiple prompt/adaptation baselines such as BiomedCoOp, KgCoOp, CoOp, CoCoOp, ProGrad, CLIP-Adapter, Tip-Adapter, LP, and LP++
- a simulated federated PromptFL-style runner for soft-prompt aggregation on a single machine

This README is written as a practical runbook for both Windows and Linux.

## What is in this repo

Important entry points:

- [`train.py`] centralized training/evaluation entry point
- [`federated/run_promptfl.py`] simulated federated runner
- [`configs/datasets`] dataset config files
- [`configs/trainers`] trainer config files
- [`scripts`] original bash helpers for Linux
- [`assets/INSTALL.md`] original installation notes
- [`assets/DATASETS.md`] dataset preparation notes
- [`assets/RUN.md`] original training/evaluation notes

## Supported methods

- BiomedCoOp
- CLIP
- CoOp
- CoCoOp
- KgCoOp
- ProGrad
- CLIP-Adapter
- Tip-Adapter
- LP
- LP++

## Supported datasets in this repo

Dataset config keys available in [`configs/datasets`]
- `btmri`
- `busi`
- `chmnist`
- `covid`
- `ctkidney`
- `dermamnist`
- `kneexray`
- `kvasir`
- `lungcolon`
- `octmnist`
- `retina`

## Environment requirements

Recommended baseline:

- Python 3.10
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA 11.8 wheels if using NVIDIA GPU

The original project was tested on Ubuntu. This repo has also been adapted to run directly on Windows without relying on the bash scripts.

## 1. Clone the repo

```bash
git clone https://github.com/HealthX-Lab/BiomedCoOp.git
cd BiomedCoOp
```

## 2. Create the environment

### Windows

Use a Python virtual environment.

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA 11.8 wheels:

```bat
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

If you want CPU-only PyTorch:

```bat
pip install torch==2.0.1 torchvision==0.15.2
```

Install repo requirements:

```bat
pip install -r requirements.txt
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

If `python` is not recognized in your shell, use:

```bat
py -3.10 train.py --help
```

or the venv Python directly:

```bat
.\.venv\Scripts\python.exe train.py --help
```

### Linux

You can use either `venv` or conda. `venv` is shown below.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA 11.8 wheels:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:

```bash
pip install torch==2.0.1 torchvision==0.15.2
```

Install repo requirements:

```bash
pip install -r requirements.txt
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

Optional conda alternative:

```bash
conda create -n biomedcoop python=3.10 -y
conda activate biomedcoop
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

## 3. Prepare the data

Dataset preparation instructions live in:

- [`assets/DATASETS.md`]

In most runs, you will pass `--root data`, so your dataset directory should look like:

```text
BiomedCoOp/
  data/
    BTMRI/
    BUSI/
    CHMNIST/
    ...
```

Example split file already present in the repo:

- [`data/BTMRI/split_BTMRI.json`]
## 4. Sanity check the install

### Windows

```bat
python train.py --help
python federated\run_promptfl.py --help
```

### Linux

```bash
python train.py --help
python federated/run_promptfl.py --help
```

If `train.py` starts importing trainers and datasets without errors, the install is mostly in good shape.

## 5. Centralized training

The main centralized entry point is [`train.py`]

General command pattern:

```text
python train.py \
  --root <data_root> \
  --seed <seed> \
  --trainer <trainer_name> \
  --dataset-config-file configs/datasets/<dataset>.yaml \
  --config-file configs/trainers/<TrainerFamily>/<setting>/<dataset>.yaml \
  --output-dir <output_dir> \
  [extra yacs options...]
```

### Example: centralized BiomedCoOp few-shot run on BTMRI

#### Windows

```bat
python train.py ^
  --root data ^
  --seed 1 ^
  --trainer BiomedCoOp_BiomedCLIP ^
  --dataset-config-file configs/datasets/btmri.yaml ^
  --config-file configs/trainers/BiomedCoOp/few_shot/btmri.yaml ^
  --output-dir output\btmri\seed1 ^
  DATASET.NUM_SHOTS 16 ^
  DATALOADER.NUM_WORKERS 0 ^
  TRAINER.BIOMEDCOOP.N_CTX 4 ^
  TRAINER.BIOMEDCOOP.CSC False ^
  TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION end
```

#### Linux

```bash
python train.py \
  --root data \
  --seed 1 \
  --trainer BiomedCoOp_BiomedCLIP \
  --dataset-config-file configs/datasets/btmri.yaml \
  --config-file configs/trainers/BiomedCoOp/few_shot/btmri.yaml \
  --output-dir output/btmri/seed1 \
  DATASET.NUM_SHOTS 16 \
  DATALOADER.NUM_WORKERS 8 \
  TRAINER.BIOMEDCOOP.N_CTX 4 \
  TRAINER.BIOMEDCOOP.CSC False \
  TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION end
```

Notes:

- `DATASET.NUM_SHOTS 16` means a 16-shot setting.
- For Windows, `DATALOADER.NUM_WORKERS 0` is the safest default.
- The BTMRI trainer config already defines BiomedCoOp-specific hyperparameters such as `SCCM_LAMBDA`, `KDSP_LAMBDA`, `TAU`, and `N_PROMPTS`.

## 6. Centralized evaluation

Use `--eval-only` with a trained output directory.

### Windows

```bat
python train.py ^
  --root data ^
  --trainer BiomedCoOp_BiomedCLIP ^
  --dataset-config-file configs/datasets/btmri.yaml ^
  --config-file configs/trainers/BiomedCoOp/few_shot/btmri.yaml ^
  --model-dir output\btmri\seed1 ^
  --load-epoch 100 ^
  --eval-only ^
  DATASET.NUM_SHOTS 16 ^
  DATALOADER.NUM_WORKERS 0
```

### Linux

```bash
python train.py \
  --root data \
  --trainer BiomedCoOp_BiomedCLIP \
  --dataset-config-file configs/datasets/btmri.yaml \
  --config-file configs/trainers/BiomedCoOp/few_shot/btmri.yaml \
  --model-dir output/btmri/seed1 \
  --load-epoch 100 \
  --eval-only \
  DATASET.NUM_SHOTS 16
```

## 7. Averaging centralized results across seeds

If you train multiple seeds, you can summarize them with:

### Windows

```bat
python parse_test_res.py output\btmri\shots_16\BiomedCoOp_BiomedCLIP\nctx4_cscFalse_ctpend --test-log
```

### Linux

```bash
python parse_test_res.py output/btmri/shots_16/BiomedCoOp_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

Adjust the path to match your output structure.

## 8. Original bash scripts on Linux

If you prefer the repo's original helper scripts, use the bash files in [`scripts`]

Example:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/few_shot.sh data btmri 16 BiomedCLIP
```

These scripts are mainly useful on Linux. On Windows, call [`train.py`] directly instead.

## 9. Simulated federated PromptFL-style training

This repo includes a single-machine federated runner:

- [`federated/run_promptfl.py`]

What it does:

- loads the centralized trainer config
- initializes the soft prompt from BiomedCoOp
- splits the few-shot training set into client partitions
- samples a subset of clients each round
- trains local prompt updates on each selected client
- averages only the soft prompt tensor with FedAvg-style aggregation
- optionally evaluates the final global prompt

The current runner federates only:

- `prompt_learner.ctx`

The backbone and text encoder remain frozen.

### Federated command example

This is the exact command shape supported by the current runner.

#### Windows

```bat
python federated\run_promptfl.py ^
  --root data ^
  --dataset btmri ^
  --trainer BiomedCoOp_BiomedCLIP ^
  --shots 16 ^
  --num-clients 5 ^
  --clients-per-round 3 ^
  --rounds 20 ^
  --local-epochs 1 ^
  --partition noniid ^
  --dirichlet-alpha 0.3 ^
  --num-workers 0 ^
  --output-dir output\promptfl ^
  --eval-final
```

#### Linux

```bash
python federated/run_promptfl.py \
  --root data \
  --dataset btmri \
  --trainer BiomedCoOp_BiomedCLIP \
  --shots 16 \
  --num-clients 5 \
  --clients-per-round 3 \
  --rounds 20 \
  --local-epochs 1 \
  --partition noniid \
  --dirichlet-alpha 0.3 \
  --num-workers 8 \
  --output-dir output/promptfl \
  --eval-final
```

### Federated output structure

Expected output shape:

```text
output/
  promptfl/
    btmri/
      split_manifest.json
      round_000/
        global_prompt.pt
      round_001/
        global_prompt.pt
        client_0/
        client_1/
        ...
      ...
      global_prompt_final.pt
      final_eval/
```

### Federated partitioning options

`federated/run_promptfl.py` currently supports:

- `--partition iid`
- `--partition noniid`

For `noniid`, the runner uses a Dirichlet split.

Recommended intuition for `--dirichlet-alpha`:

- `0.1`: very non-IID
- `0.3`: strongly non-IID
- `0.5`: moderately non-IID
- `1.0`: mildly non-IID
- `10.0`: close to IID

### Weighted vs unweighted aggregation

Default behavior is sample-weighted averaging.

To switch to a plain mean over selected clients:

#### Windows

```bat
python federated\run_promptfl.py ... --unweighted
```

#### Linux

```bash
python federated/run_promptfl.py ... --unweighted
```

## 10. Common trainer names

Examples you can pass to `--trainer`:

- `BiomedCoOp_BiomedCLIP`
- `BiomedCoOp_CLIP`
- `BiomedCoOp_PubMedCLIP`
- `BiomedCoOp_PMCCLIP`
- `KgCoOp_BiomedCLIP`
- `CoOp_BiomedCLIP`
- `CoCoOp_BiomedCLIP`
- `ProGrad_BiomedCLIP`

The exact available implementations live under [`trainers`]

## 11. Useful config files

Examples:

- dataset config: [`configs/datasets/btmri.yaml`]
- BiomedCoOp few-shot config: [`configs/trainers/BiomedCoOp/few_shot/btmri.yaml`]
- KgCoOp few-shot config: [`configs/trainers/KgCoOp/few_shot/btmri.yaml`]
Common runtime overrides:

- `DATASET.NUM_SHOTS 16`
- `DATALOADER.NUM_WORKERS 0`
- `DATALOADER.TRAIN_X.BATCH_SIZE 2`
- `OPTIM.MAX_EPOCH 10`
- `TRAINER.BIOMEDCOOP.N_CTX 4`

## 12. Windows-specific notes

- Use `^` for line continuation in Command Prompt.
- In PowerShell, either place the command on one line or use the backtick continuation character.
- If dataloader multiprocessing causes issues, set:

```text
DATALOADER.NUM_WORKERS 0
```

- If you hit GPU memory issues, reduce batch size:

```text
DATALOADER.TRAIN_X.BATCH_SIZE 2
```

- If `python` is not available, use `py -3.10`.
- Paths in commands can use backslashes, for example `output\promptfl\btmri`.

## 13. Linux-specific notes

- You can use the original `.sh` helper scripts directly.
- `CUDA_VISIBLE_DEVICES=0` is the easiest way to pick a GPU for centralized runs.
- If your machine has enough RAM and no multiprocessing issues, `DATALOADER.NUM_WORKERS 8` is a reasonable starting point.

## 14. Troubleshooting

### `python` is not recognized

Windows:

```bat
py -3.10 train.py --help
```

or:

```bat
.\.venv\Scripts\python.exe train.py --help
```

### `ModuleNotFoundError: dassl`

You likely skipped the Dassl installation step:

```bash
cd Dassl.pytorch
python setup.py develop
cd ..
```

### CUDA out of memory

Reduce batch size and workers:

```text
DATALOADER.TRAIN_X.BATCH_SIZE 2
DATALOADER.NUM_WORKERS 0
```

For the federated runner, also reduce:

- `--clients-per-round`
- `--local-epochs`

### Windows dataloader hangs or crashes

Use:

```text
DATALOADER.NUM_WORKERS 0
```

### Hugging Face / model download issues

Retry after the first run, and make sure the machine can reach Hugging Face. If needed, pre-populate the cache in the same environment before launching long experiments.

### Bash scripts do not work on Windows

That is expected unless you are using WSL or Git Bash. On Windows, call [`train.py`] or [`federated/run_promptfl.py`] directly.

## 15. Suggested first runs

If you want the quickest sanity check:

1. Centralized few-shot BiomedCoOp on BTMRI with `DATASET.NUM_SHOTS 16`
2. Federated BTMRI run with:
   - `--num-clients 3`
   - `--clients-per-round 2`
   - `--rounds 5`
   - `--local-epochs 1`

That gets you to a working baseline quickly before scaling to larger runs.

## 16. Original paper and assets

- Paper: [https://arxiv.org/abs/2411.15232](https://arxiv.org/abs/2411.15232)
- Installation notes: [`assets/INSTALL.md`]
- Dataset notes: [`assets/DATASETS.md`]
- Original run notes: [`assets/RUN.md`]

```
