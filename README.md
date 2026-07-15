# What Drives Neural Pairwise Bug Verification? A Multilingual Empirical Study of Evaluation Leakage, Representation Learning, and Graph Structure

This repository contains the replication package for a large-scale empirical
study of neural pairwise bug verification.

The study investigates how different components contribute to verification
performance, including:

- pretrained code representations,
- learned pairwise comparison functions,
- graph-based program representations,
- heterogeneous edge structures,
- and evaluation protocol design.

The experimental framework includes **SiameseGAT**, a Siamese graph attention
framework over heterogeneous code graphs (sequential, control-flow, and
data-flow edges) with frozen GraphCodeBERT node features. SiameseGAT is used as
an experimental instrument for controlled ablation studies rather than as a
standalone proposed architecture.

Experiments cover **64,274 program pairs** across six programming languages
(**Python, C++, Java, C, Ruby, and JavaScript**) from **Project CodeNet**, with
out-of-distribution evaluation on **HumanEvalFix**.

The goal of this work is to systematically analyze what current neural
pairwise verification systems learn and how performance depends on
representations, comparison functions, graph structures, and evaluation
methodology.

---

## Data availability

- Raw data is public:
  - [Project CodeNet](https://github.com/IBM/Project_CodeNet) (Apache 2.0)
  - [HumanEvalFix / OctoPack](https://github.com/bigcode-project/octopack) (MIT)

- Constructed pair lists, precomputed graph tensors, and trained checkpoints
  (~300 GB) will be deposited on Zenodo with a permanent DOI upon acceptance.

- All construction and training code is available here. The complete dataset
  of program pairs can be reproduced from the public raw sources using the
  provided pipeline scripts.

---

## Experimental framework

The repository implements the experimental framework used to study the effects
of:

1. Evaluation protocol design
2. Pretrained representations
3. Learned pairwise comparison
4. Graph structure and edge types
5. Out-of-distribution transfer

The framework uses:

- heterogeneous code graphs containing sequential, control-flow, and data-flow
  edges,
- frozen GraphCodeBERT node representations,
- Siamese graph encoding,
- pairwise comparison and classification.

The architecture is intentionally designed for controlled component analysis,
allowing individual factors to be removed or replaced through ablation studies.

---

## Setup

Experiments were conducted using:

- PyTorch 2.1
- PyTorch Geometric 2.4
- tree-sitter 0.20.1
- HuggingFace Transformers
- `microsoft/graphcodebert-base`

Hardware:

- 2 × NVIDIA Tesla P40 GPUs (23 GB each)

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/download_datasets.py` | Raw CodeNet / HumanEvalFix data |
| 2 | `scripts/build_pairs.py` | AC/WA pair lists (`data/processed/`) |
| 3 | `scripts/build_graphs.py` | PyG graphs + GraphCodeBERT embeddings (`data/graphs/`) |
| 4 | `scripts/train.py` | 5-fold GroupKFold CV, per-language evaluation |
| 5 | `scripts/ablation_edges.py`, `run_ablation.sh` | Edge-type ablation experiments |
| 6 | `scripts/train_gcbert_meanpool.py`, `scripts/baseline_cls.py` | Representation and comparator baselines |
| 7 | `scripts/ablation_edit_distance.py` | Near-identical pair construct-validity analysis |
| 8 | `scripts/llm_baselines.py` | LLM judge baselines (DeepSeek-V3 / Qwen2.5-Coder) |

Paths default to the repository layout above. Adjust data roots in:

```text
configs/config.yaml

## Contact

Ologun S. Babatunde, UESTC. Issues welcome.
