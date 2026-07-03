# Embodied LLaVA with 3D Spatial Reasoning

## Project Overview

This project is a prototype framework for embodied AI built on top of a multimodal large language model, specifically LLaVA-1.5. The goal is to improve multimodal reasoning by introducing explicit 3D spatial information into the model.

The framework is designed for two embodied tasks:

1. 3D Spatial Question Answering on ScanQA
2. Language-guided Navigation on RoboTHOR

At the current stage, the implementation focuses on the ScanQA-style question answering pathway and validates the core multimodal pipeline.

---

## Core Idea

The main idea of this project is to inject structured 3D scene information into a multimodal LLM.

Instead of relying only on image and text input, the model also receives:

- 3D object bounding boxes `(x, y, z, w, h, d)`
- semantic category labels

These are encoded by a lightweight 3D spatial encoder and fused with visual information before being passed to the LLaVA backbone.

---

## Current Implementation

The following components have been implemented:

- 4-bit loading of `llava-hf/llava-1.5-7b-hf`
- `Spatial3DEncoder` for object-level 3D representation
- `QFormerFusion` for multimodal fusion
- `EmbodiedLLaVA` wrapper model
- toy ScanQA-style dataset
- forward pass validation
- QA loss validation
- toy end-to-end training loop

The current codebase validates that the architecture is runnable and trainable.

---

## Project Structure

```text
src/
  data/
  encoders/
  fusion/
  models/
  train/
  utils/

scripts/
  test_spatial_3d_encoder.py
  test_q_former_fusion.py
  test_toy_scanqa_forward.py
  train_toy_scanqa.py

```
## Environment
Python environment was created with Conda.
Main dependencies include:
PyTorch
transformers
peft
accelerate
bitsandbytes

---
```
## How to run

1. Activate environment
conda activate llava-qlora

2. Verify 4-bit LLaVA loading
python verify_llava_4bit.py

3. Run toy ScanQA forward test
python -m scripts.test_toy_scanqa_forward

4. Run toy ScanQA training loop
python -m scripts.train_toy_scanqa

```
## Current Status

This repository currently contains a prototype-stage implementation.
The toy ScanQA pipeline is used only for engineering validation.
It shows that the following stages are connected correctly:
data loading
spatial encoding
multimodal fusion
LLaVA forward pass
loss computation
backpropagation
optimizer update

This is not yet the final benchmark result on the real ScanQA dataset.
Next Steps

The next development steps are:
replace toy ScanQA with real ScanQA data
connect real visual encoder outputs
implement formal ScanQA training and evaluation
extend the framework to RoboTHOR navigation
compare multiple fusion strategies in ablation experiments