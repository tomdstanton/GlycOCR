# GlycOCR Project Justfile
# Run `just` to see all available commands

set shell := ["bash", "-uc"]

# Show available commands
default:
    @just --list

# Clean Python virtual environments
clean:
    rm -rf site
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +

install: clean
    uv sync

# Run the test suite
test: install
    uv run pytest tests/

# Format all Python code
fmt:
    uvx ruff format .

# Check if code is formatted without modifying files
fmt-check:
    uvx ruff format --check .

# Lint Python code and auto-fix safe errors
lint:
    uvx ruff check --fix .

# Static type-check Python code
type-check:
    uvx ty check .

# Run all quality checks at once (ideal for local pre-commit testing)
check-all: fmt-check lint type-check

# Run the full CI pipeline locally (format check, lint, test)
ci: check-all test

# Build the Python package
build:
    uv build

# Publish the Python package to PyPI
publish:
    uv publish

# --- GlycOCR CLI Commands ---

# 1. Fetch real glycans and generate dataset list
fetch-dataset:
    uv run glycocr fetch-dataset --output data/dataset_iupac.txt --synthetic-ratio 0.5 --verbose

# 2. Synthesize SNFG images from the generated IUPAC list
synthesize:
    uv run glycocr synthesize --iupac-list data/dataset_iupac.txt --out-dir data/images/ --verbose

# 3. Train the model on the generated dataset (WARNING: Requires GPU / HPC)
train:
    uv run glycocr train --data-dir data/images/ --output-dir models/checkpoints/ --epochs 3 --batch-size 4 --lr 0.0005 --verbose

# 4. Predict an IUPAC string from an SNFG image
predict IMAGE_PATH:
    uv run glycocr predict --image {{IMAGE_PATH}} --verbose

# 5. Scan a PDF for SNFG images and stream out IUPAC strings
scan PDF_PATH OUTPUT_FILE="":
    uv run glycocr scan --pdf {{PDF_PATH}} {{if OUTPUT_FILE != "" { "--output " + OUTPUT_FILE } else { "" }}} --verbose

# --- HPC Scripts ---

# Submit training job to Monash MASSIVE M3 SLURM cluster
submit-train:
    sbatch train.slurm
