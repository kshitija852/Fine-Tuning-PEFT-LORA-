# Fine-Tuning with PEFT + LoRA on Hugging Face Transformers

This repository demonstrates **Parameter-Efficient Fine-Tuning (PEFT)** using **Low-Rank Adaptation (LoRA)** with a pre-trained DistilBERT model for **text classification** on the Rotten Tomatoes dataset.

It’s a minimal and easy-to-understand example of fine-tuning only a small number of parameters instead of the entire model — resulting in **faster training and reduced GPU usage**.

---

## Overview

| Component | Description |
|------------|-------------|
| **Model** | `distilbert-base-uncased` |
| **Technique** | LoRA (Low-Rank Adaptation) |
| **Dataset** | Rotten Tomatoes (Sentiment Classification) |
| **Frameworks** | Hugging Face Transformers + PEFT + PyTorch |
| **Task** | 3-class Sentiment Classification: Positive / Negative / Neutral |

---

## What is PEFT and LoRA?

- **PEFT (Parameter Efficient Fine-Tuning)** allows training large models efficiently by updating only a small portion of their parameters.
- **LoRA (Low-Rank Adaptation)** inserts small, trainable adapter layers into the transformer architecture — drastically reducing memory requirements while maintaining performance.

---

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/kshitija852/Fine-Tuning-PEFT-LORA.git
cd Fine-Tuning-PEFT-LORA
