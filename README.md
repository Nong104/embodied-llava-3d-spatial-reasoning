# Embodying Language Models using Multimodal LLM

Master's thesis project: a controlled ablation framework for grounding a frozen multimodal LLM (LLaVA-1.5-7B) in 3D spatial perception, applied to two embodied tasks: 3D spatial question answering (ScanQA) and language-guided navigation (RoboTHOR).

## Project Overview

Current multimodal LLMs can describe a scene but cannot act within it, and lack explicit mechanisms for 3D spatial reasoning. This project builds a unified framework that shares a single frozen LLaVA-1.5 backbone across two tasks, with only a lightweight spatial encoder and an interchangeable fusion module trained. Four fusion strategies — Early Fusion, Q-Former (cross-modal attention), Late Fusion, and Mixture-of-Experts — are compared under identical, controlled conditions.

## Key Results

### ScanQA Fusion Strategy Ablation (three random seeds, LoRA fine-tuning, scene-disjoint split)

| Strategy | CIDEr (mean ± SD) | EM (mean) | Trainable Params | Peak GPU Memory |
|---|---|---|---|---|
| Early | 0.219 ± 0.138 | 0.011 | 12,030,464 | 6.74 GB |
| Q-Former | 0.223 ± 0.092 | 0.011 | 230,314,496 | 10.36 GB |
| Late | 0.341 ± 0.074 | 0.011 | 62,374,400 | 7.61 GB |
| MoE | 0.266 ± 0.057 | 0.011 | 146,297,348 | 9.07 GB |

All six pairwise comparisons were tested with a paired bootstrap significance test (10,000 resamples); none reached p < 0.05, indicating the current three-seed sample is not yet large enough to declare one strategy definitively superior. An earlier single-seed run had shown Q-Former far ahead (CIDEr 0.612); the multi-seed result shows this was largely due to a favourable random seed rather than a stable property of the architecture.

### Zero-Shot Baseline Comparison

| Model | CIDEr |
|---|---|
| Zero-shot InstructBLIP | 0.117 |
| Zero-shot LLaVA | 0.158 |
| Early Fusion | 0.219 |
| Q-Former | 0.223 |
| MoE | 0.266 |
| Late Fusion | 0.341 |

All four fusion strategies outperform both zero-shot baselines, indicating the spatial fusion approach provides a measurable benefit over a general-purpose multimodal model with no dedicated spatial grounding.

### RoboTHOR Navigation (three random seeds, real CLIP visual encoding, imitation learning)

| Seed | Success Rate |
|---|---|
| 42 | 0.000 |
| 123 | 0.333 |
| 2024 | 0.000 |

Mean SR 0.111, SD 0.192. Diagnostic checks confirmed the navigation head correctly receives scene information; the low, highly variable success rate is attributed to insufficient training episodes rather than a structural fault.

## Repository Structure

```text
src/
  data/          # ScanQA/ScanNet/RoboTHOR dataset loaders and the ScanNet mesh parser
  encoders/      # Spatial3DEncoder
  fusion/        # EarlyFusion, QFormerFusion, LateFusion, MoEFusion
  models/        # EmbodiedLLaVA wrapper, LLaVA 4-bit loader, LoRA attachment, CLIP vision encoding, text generation
  navigation/    # NavigationHead, expert-path-to-action conversion

scripts/
  train_single_strategy.py         # ScanQA fusion strategy training + evaluation (single seed/strategy per run)
  train_and_eval_navigation.py     # RoboTHOR imitation learning training + evaluation
  eval_zeroshot_llava.py           # Zero-shot LLaVA baseline
  eval_zeroshot_instructblip.py    # Zero-shot InstructBLIP baseline
  scannet_parser.py                # Parses real ScanNet mesh/segmentation/aggregation files into object bounding boxes and colours

data/
  scanqa_raw/       # Official ScanQA question-answer JSON files
  scannet_raw/       # Real ScanNet scene files (not included in this repository; see Data Setup below)
```

## Data Setup

This repository does not include the raw ScanNet scene files (`data/scannet_raw/scans/`), since they total several gigabytes and require an application through the official ScanNet data agreement. To reproduce the experiments:

1. Apply for ScanNet access at [github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet) and obtain the download script.
2. Download scene files (`.aggregation.json`, `_vh_clean_2.ply`, `_vh_clean_2.0.010000.segs.json`) into `data/scannet_raw/scans/<scene_id>/`.
3. Download the official ScanQA annotation files into `data/scanqa_raw/` from [github.com/ATR-DBI/ScanQA](https://github.com/ATR-DBI/ScanQA).
4. RoboTHOR runs require AI2-THOR, which is Linux-only; this project was developed with the training pipeline on Windows and the navigation environment on WSL2/Ubuntu.

## Environment

Two environments were used across this project, due to LLaVA 4-bit quantisation being most stable on Windows and AI2-THOR requiring Linux:

- **Windows (ScanQA track):** Conda env, PyTorch with CUDA, `transformers`, `peft`, `bitsandbytes`, `accelerate`.
- **WSL2/Ubuntu (RoboTHOR track):** `ai2thor`, `plyfile`, PyTorch, `transformers`.

## How to Run

**ScanQA fusion strategy ablation** (repeat with different `--seed` values for multi-seed results):

```bash
python -m scripts.train_single_strategy --strategy early --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy qformer --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy late --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy moe --steps 300 --seed 42 --batch-size 2
```

**Zero-shot baselines:**

```bash
python -m scripts.eval_zeroshot_llava
python -m scripts.eval_zeroshot_instructblip
```

**RoboTHOR navigation** (in WSL2/Ubuntu):

```bash
python3 -m scripts.train_and_eval_navigation --seed 42 --episodes 100
```

## Known Limitations

- The paired bootstrap significance test currently lacks statistical power with only three seeds; all pairwise strategy comparisons are not yet significant at p < 0.05.
- Two of the five originally planned baseline comparisons (a dedicated ScanQA baseline model and NavGPT) were not completed, as both depend on CUDA/C++ compiled extensions not reliably achievable on the Windows environment used for this project within the available time.
- RoboTHOR results reflect a training scale of 50–100 episodes; success rate is low and highly variable, and is expected to improve substantially with a larger training budget.
- The real ScanNet coordinate data covers a subset of scenes with genuine ScanQA annotations (135–274 depending on the split), not the full 800-scene ScanQA corpus.