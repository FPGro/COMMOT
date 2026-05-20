# COMMOT

Screening cell–cell communication in spatial transcriptomics via
collective optimal transport.

This repository is a maintained fork of
[zcang/COMMOT](https://github.com/zcang/COMMOT), incorporating the
performance optimizations from
[Zaoqu-Liu/COMMOT](https://github.com/Zaoqu-Liu/COMMOT) and updating
the codebase to work with modern Python and package versions.

## What changed from the original

| Area | Change |
|------|--------|
| **Performance** | 2–3× faster via batch gene extraction and parallel COT (Zaoqu-Liu) |
| **Python** | Requires Python ≥ 3.10 (was 3.7) |
| **anndata** | Compatible with anndata 0.10–0.12 (was pinned to 0.7.6) |
| **Dependencies** | All pins modernized; `pysal` → `libpysal` |
| **Bug fixes** | Missing functions restored; pandas/scipy/networkx deprecations fixed |

## Installation

```bash
git clone https://github.com/YOUR_USER/COMMOT.git
cd COMMOT
pip install .
```

For downstream analysis functions (`group_cell_communication`):
```bash
pip install \".[downstream]\"
# karateclub must be installed separately from GitHub:
pip install git+https://github.com/benedekrozemberczki/karateclub.git@cb46a91
```

> **Note**: Do not use `pip install commot` — that installs the original
> unpatched version from PyPI.

## Usage

The API is fully compatible with the original. See the
[official documentation](https://commot.readthedocs.io/en/latest/) for
complete usage examples.

```python
import commot as ct
import scanpy as sc

adata = sc.read_h5ad(\"your_data.h5ad\")

df_ligrec = ct.pp.ligand_receptor_database(database='CellChat', species='human')

ct.tl.spatial_communication(
    adata,
    database_name='CellChat',
    df_ligrec=df_ligrec,
    dis_thr=200,
    heteromeric=True,
    n_jobs=-1   # parallelization (new parameter)
)
```

## Citation

If you use this software, please cite the original paper:

Cang, Z., Zhao, Y., Almet, A.A. et al. Screening cell–cell communication
in spatial transcriptomics via collective optimal transport. *Nat Methods*
20, 218–228 (2023). https://doi.org/10.1038/s41592-022-01728-4

## License

MIT License — original copyright Zixuan Cang (2022), see `LICENSE.md`.