# TriM-DTA

TriM-DTA is a tri-modal framework for drug--target affinity prediction. It combines molecular and protein sequence representations with topological and geometric graph features.

## Setup

Use Python 3.8+ with a CUDA-enabled PyTorch installation, then install the remaining packages:

```bash
pip install -r requirements.txt
```

The code expects the following local resources, which are intentionally not tracked in this repository: `davis/` or `kiba/` data splits and processed CSVs, `target_contact_map_<dataset>/` contact maps, `Vocab/` vocabulary pickles, and PyG structure graphs (for example `davis/pyg_8/`). Generate structure graphs from PDB/SDF inputs with helpers in `process.py`.

## Train

The default training entry point uses the Davis dataset and CUDA device 0:

```bash
python main.py --dim 128 --epoch 100 --batch_size 100 --lr 1e-4
```

For the reproducible KIBA scaling experiment:

```bash
python trim_dta_experiment.py --dataset kiba --mode default --dim 128
```

`scripts/run.sh` contains example Davis hyperparameter runs, and `scripts/run_kiba_scalability.sh` runs the KIBA scale sweep.

## Repository layout

- `main.py`: TriM-DTA model definition and training loop.
- `dataset.py`, `model.py`, `egnn.py`, `moe.py`: data and model components.
- `process.py`: PDB/SDF to PyTorch Geometric preprocessing helpers.
- `trim_dta_experiment.py`: KIBA experiment with timing and memory reporting.

## Citation

Citation information will be added with the accompanying manuscript.
