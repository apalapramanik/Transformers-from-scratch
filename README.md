
# Transformer From Scratch

This repository implements a **GPT-style Transformer language model from scratch**, with an emphasis on **conceptual clarity, correctness, and transparency**.

The goal of this project is to understand how Transformers work by explicitly implementing each core component, rather than relying on high-level library abstractions.

---

## Overview

The model is trained as a **character-level autoregressive language model** on the **WikiText-2 (raw)** dataset.
Given a sequence of characters, the model learns to predict the next character using **causal self-attention**.

The implementation focuses on:

* Explicit attention mechanics
* Clear data flow
* Stable and debuggable training

---

## Model Architecture

The model follows a GPT-style Transformer architecture:

```
tokens
 → token embedding
 → sinusoidal positional encoding
 → N × Transformer blocks
 → layer normalization
 → linear output head
 → next-character prediction
```

### Transformer Block

Each Transformer block consists of:

1. Multi-head self-attention with causal masking
2. Residual connection followed by layer normalization
3. Feed-forward network
4. Residual connection followed by layer normalization

---

## Features

### Core Components

* Scaled dot-product attention
* Multi-head self-attention
* Causal masking (autoregressive)
* Sinusoidal positional encoding
* Residual connections
* Layer normalization
* Feed-forward network

### Training Pipeline

* Character-level dataset loader
* Sliding-window sequence generation
* Teacher forcing
* Cross-entropy loss
* Adam optimizer
* Gradient clipping
* Learning-rate scheduling
* Validation loop
* Checkpoint saving
* Text generation
* Training and validation loss plotting

### Platform Support

* Apple Silicon (MPS)
* CPU fallback
* Automatic device selection

---

## Repository Structure

```
transformer-from-scratch/
├── train.py
├── requirements.txt
├── checkpoints/
├── loss_curve.png
├── data/
│   └── wikitext-2-raw/
│       ├── wiki.train.raw
│       ├── wiki.valid.raw
│       └── wiki.test.raw
├── src/
│   ├── dataset.py
│   └── model/
│       ├── attention.py
│       ├── multi_head_attention.py
│       ├── transformer_block.py
│       ├── positional_encoding.py
│       └── transformer_model.py
└── README.md
```

---

## Dataset

The model is trained on **WikiText-2 (raw version)** using **character-level language modeling**.

* No tokenization is used
* Each training example is a fixed-length character sequence
* Targets are the same sequence shifted by one character

Approximate dataset sizes:

* Training: ~12M characters
* Validation: ~1M characters
* Test: ~1M characters

---

## Requirements

### System Requirements

* macOS, Linux, or Windows
* Apple Silicon supported via MPS
* CPU-only execution supported

### Software Requirements

* **Python ≥ 3.10**

All Python dependencies are listed in `requirements.txt`.

---

## Installation

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Download

The dataset is downloaded using the Hugging Face `datasets` API and saved locally as plain text.

```python
from datasets import load_dataset
import os

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
os.makedirs("data/wikitext-2-raw", exist_ok=True)

with open("data/wikitext-2-raw/wiki.train.raw", "w") as f:
    f.write("\n".join(dataset["train"]["text"]))

with open("data/wikitext-2-raw/wiki.valid.raw", "w") as f:
    f.write("\n".join(dataset["validation"]["text"]))

with open("data/wikitext-2-raw/wiki.test.raw", "w") as f:
    f.write("\n".join(dataset["test"]["text"]))
```

---

## Training

Run training from the repository root:

```bash
python train.py
```

During training, the script:

* Prints batch-level progress
* Reports training and validation loss per epoch
* Saves model checkpoints
* Generates sample text after each epoch
* Saves a training/validation loss plot (`loss_curve.png`)

---

## Text Generation

Text generation is performed using:

* Autoregressive decoding
* Temperature-scaled sampling
* Multinomial token selection

Generated samples provide a qualitative check on model learning and stability.




