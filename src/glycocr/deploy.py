"""Model deployment API for Hugging Face Hub."""

import os
import logging
from huggingface_hub import HfApi
from pathlib import Path

logger = logging.getLogger(__name__)

def deploy_to_huggingface(repo_id: str, model_path: str, token: str = None) -> None:
    """Deploy a model directory or file to a Hugging Face Hub repository.
    
    Args:
        repo_id: Target repository ID (e.g., 'username/glycocr')
        model_path: Path to the model file or directory
        token: HF API token. If None, uses HF_TOKEN environment variable.
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable or token argument must be provided.")
        
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    api = HfApi(token=token)
    
    logger.info(f"Creating repository {repo_id} (if it doesn't exist)...")
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    logger.info(f"Uploading {model_path} to {repo_id}...")
    if model_path_obj.is_dir():
        api.upload_folder(
            folder_path=str(model_path_obj),
            repo_id=repo_id,
            commit_message="Deploy GlycOCR model",
        )
    else:
        api.upload_file(
            path_or_fileobj=str(model_path_obj),
            path_in_repo=model_path_obj.name,
            repo_id=repo_id,
            commit_message="Deploy GlycOCR model file",
        )
    logger.info("Deployment successful!")
