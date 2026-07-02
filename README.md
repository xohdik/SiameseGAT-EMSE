# What Drives Neural Pairwise Bug Verification and Why It Fails to Transfer?

Replication package for the EMSE submission:
**"What Drives Neural Pairwise Bug Verification and Why It Fails to Transfer? A Multilingual Ablation of Embeddings, Graph Attention, and Edge Types"**

SiameseGAT: a Siamese graph attention network over heterogeneous code graphs
(sequential, control-flow, and data-flow edges) with frozen GraphCodeBERT node
features, evaluated on 64,274 program pairs across six languages (Python, C++,
Java, C, Ruby, JavaScript) from Project CodeNet, with out-of-distribution
evaluation on HumanEvalFix.

## Data availability

- Raw data is public: [Project CodeNet](https://github.com/IBM/Project_CodeNet) (Apache 2.0)
  and [HumanEvalFix / OctoPack](https://github.com/bigcode-project/octopack) (MIT).
- Constructed pair lists, precomputed graph tensors, and trained checkpoints
  (~300 GB) will be deposited on Zenodo with a permanent DOI upon acceptance.
- All construction and training code is available here now; the full dataset is
  reproducible from the raw sources with the pipeline below.

## Setup
PyTorch 2.1, PyTorch Geometric 2.4, tree-sitter 0.20.1, transformers
(microsoft/graphcodebert-base). Experiments ran on 2x NVIDIA Tesla P40 (23 GB).

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/download_datasets.py` | raw CodeNet / HumanEvalFix |
| 2 | `scripts/build_pairs.py` | AC/WA pair lists (`data/processed/`) |
| 3 | `scripts/build_graphs.py` | PyG graphs + GCBERT embeddings (`data/graphs/`) |
| 4 | `scripts/train.py` | 5-fold GroupKFold CV, per-language (Tables 9-10) |
| 5 | `scripts/ablation_edges.py`, `run_ablation.sh` | edge-type ablation (Table 13) |
| 6 | `scripts/train_gcbert_meanpool.py`, `scripts/baseline_cls.py` | component ablation baselines (Tables 11-12) |
| 7 | `scripts/ablation_edit_distance.py` | near-identical-pair construct-validity test |
| 8 | `scripts/llm_baselines.py` | DeepSeek-V3 / Qwen2.5-Coder judge baselines (Table 16) |

Paths default to the layout above; adjust data roots in `configs/config.yaml`.

## Contact

Ologun S. Babatunde, UESTC. Issues welcome.
