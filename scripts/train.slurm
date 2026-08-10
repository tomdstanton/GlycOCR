#!/bin/bash
#SBATCH --job-name=glycocr-train
#SBATCH --account=<YOUR_PROJECT_CODE>
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Starting GlycOCR training job..."
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# 2. Setup the environment with uv
# If uv is not installed on M3 globally, install it locally:
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Install dependencies 
# We sync all dependencies, including the [train] group we set up earlier
echo "Syncing dependencies with uv..."
uv sync --extra train

# 4. Synthesize the dataset (if not already synthesized)
# We synthesize a large dataset for training. 
# You may want to adjust the size depending on your compute limits.
DATA_DIR="/scratch/kr58/tom/glycocr_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Fetching and generating IUPAC sequences..."
    uv run glycocr prep fetch iupacs.txt --synthetic-ratio 1.0
    
    echo "Synthesizing training data..."
    uv run glycocr prep synthesize iupacs.txt $DATA_DIR
else
    echo "Training data already exists at $DATA_DIR"
fi

# 5. Run the training pipeline
echo "Kicking off model training..."
uv run glycocr train $DATA_DIR models/glycocr_paligemma2

echo "Training complete!"
