# PPI-Transformer

Transformer-based protein-protein interaction (PPI) classifier that combines HuRI interaction labels with sequence embeddings from ESM. Training and evaluation are orchestrated with PyTorch Lightning and logged to Weights & Biases (W&B).

## Quick Start
- **Clone & enter**
  ```bash
  git clone <repo-url>
  cd PPI-Transformer
  ```
- **Create an environment** (example with conda)
  ```bash
  conda create -n ppi-transformer python=3.10
  conda activate ppi-transformer
  ```
- **Install dependencies**
  ```bash
  pip install torch pytorch-lightning pandas transformers tdc wandb
  ```
- **Authenticate W&B (optional but recommended)**
  ```bash
  wandb login
  ```

## Data Preparation Pipeline
1. **Download HuRI splits**
   ```bash
   python dataset/download_huri.py
   ```
   Writes `data/huri_train.csv`, `data/huri_val.csv`, and `data/huri_test.csv` using the TDC `PPI` interface.
2. **Enumerate unique proteins**
   ```bash
   python dataset/unique_protein.py
   ```
   Aggregates all proteins into `data/huri_unique_proteins.csv` with columns `Protein_ID` and `Protein`.
3. **Generate ESM embeddings** *(GPU strongly recommended)*
   ```bash
   python dataset/esm_generator.py
   ```
   Uses `facebook/esm2_t6_8M_UR50D` from Hugging Face Transformers to create one `.pt` file per protein under `data/esm_embeddings/`. Ensure the directory exists and you have sufficient disk space.
   
   For SLURM clusters, use `dataset/esm_gen.sh` which activates the `PPI-Transformer` conda environment before running the generator.

> **Expected CSV schema**: `Protein1_ID`, `Protein2_ID`, `Protein1`, `Protein2`, `Y`. Labels are binary (0/1). Embedding files must be named `<Protein_ID>.pt` to match the loader.

## Training & Evaluation
- **Single run (local GPU)**
  ```bash
  python train.py
  ```
  Loads data via `dataset/ppiDataModule.py`, trains `model/ppiPredictor.py` for 10 epochs, and immediately evaluates on the test split. Metrics are logged to W&B.
- **Cluster job submission**
  Submit `train.sh` on SLURM after adjusting resources and environment paths. The script activates conda and runs the same training entry point.

PyTorch Lightning will default to the first visible GPU. To run on CPU or multiple GPUs, adjust `Trainer` arguments in `train.py` (e.g., change `accelerator` or `devices`).

## Repository Layout
```
train.py                # Lightning training/eval loop
train.sh                # SLURM helper script
 dataset/
   download_huri.py     # Pull HuRI splits via TDC
   unique_protein.py    # Build unique protein list
   esm_generator.py     # Produce ESM sequence embeddings
   collate_fn.py        # Batch collation with padding
   ppiDataset.py        # Dataset wrapper loading embeddings
   ppiDataModule.py     # Lightning DataModule for HuRI splits
   esm_gen.sh           # SLURM helper for embedding generation
 model/
   ppiPredictor.py      # LightningModule defining the classifier
   modules/ppiTransformer.py  # Transformer encoder over paired seqs
 data/                  # CSV splits and generated embeddings (gitignored)
```

## Configuration Notes
- **Hyperparameters**: tweak `batch_size` in `ppiDataModule`, learning rate and loss in `ppiPredictor`, or update the `Trainer` settings (epochs, precision, callbacks) in `train.py`.
- **Model architecture**: `ppiTransformer` concatenates padded embeddings, applies a single-layer transformer encoder (`d_model=320`, `nhead=8`), mean-pools, and predicts interaction probability via a sigmoid-activated linear head.
- **Custom embeddings**: drop-in replacements are possible by modifying `dataset/ppiDataset.py` to load alternate files or dimensionalities.

## Logging & Experiment Tracking
- Training runs write to W&B project `PPI-Transformer`. Update the project name in `train.py` if desired.
- To disable W&B entirely, remove the logger instantiation and `wandb.finish()` call.

## Troubleshooting
- **Missing embeddings**: ensure every protein ID referenced in the CSV splits has a matching `.pt` tensor in `data/esm_embeddings/`.
- **Token length**: sequences over 1,024 residues are truncated by the tokenizer; adjust `max_length` in `dataset/esm_generator.py` if you switch to a larger ESM backbone.
- **CPU-only setups**: comment out or change the `accelerator="gpu"` argument when no CUDA device is available.

> No license file is currently provided; confirm usage terms before redistribution.
