# GlycOCR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

GlycOCR is an Optical Character Recognition tool for Symbol Nomenclature for Glycans (SNFG) diagram images. It translates 2D SNFG diagrams into standardized IUPAC-condensed strings using Google's PaliGemma 2 (google/paligemma2-3b-pt-448) vision-language architecture.

## Background

Extracting machine-readable glycan structures from literature is often hampered by the prevalence of visual SNFG diagrams rather than text annotations. GlycOCR leverages a fine-tuned vision-language model to directly read these images and predict their corresponding IUPAC-condensed representation.

## Installation

Install directly using `uv` or `pip`:

```bash
uv pip install glycocr
```

If you plan to train or fine-tune GlycOCR from scratch, install the optional training dependencies:

```bash
uv pip install glycocr[train]
```

## CLI Usage

GlycOCR provides a unified CLI for inference, training, data synthesis, and deployment. The CLI now supports standard `-v/--version` and `-V/--verbose` flags, and utilizes standard positional arguments for input and output (with `-` defaulting to stdin/stdout).

### Inference

Run inference on an SNFG image to output its parsed IUPAC string in a JSON-lines format (using high-performance `orjson` serialization):

```bash
glycocr infer image.png output.json
```

### Data Preparation

Fetch a list of real glycans from the `glycowork` database, and augment them with randomized synthetic variants:

```bash
glycocr prep fetch sequences.txt --synthetic-ratio 0.5
```

Generate synthetic SNFG images from the IUPAC list, creating a highly optimized PyTorch binary Structure-of-Arrays (SoA) dataset:

```bash
glycocr prep synthesize sequences.txt ./binary_dataset
```

### Training

Start fine-tuning the model on your generated binary dataset:

```bash
glycocr train ./binary_dataset ./output_model_dir --epochs 3 --batch-size 4
```

### Deployment

Deploy a trained model securely to the Hugging Face Hub:

```bash
glycocr deploy <username>/glycocr ./output_model_dir
```
