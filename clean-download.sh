#!/bin/bash

# Set variables
CACHE_DIR=".cache/huggingface"
MODEL_ID="stabilityai/stable-diffusion-3.5-large"

echo "Cleaning up cache directory..."
rm -rf "$CACHE_DIR/hub/models--stabilityai--stable-diffusion-3.5-large"

# Create Python download script
cat > download_script.py << 'EOL'
import os
import time
import sys
from huggingface_hub import snapshot_download, HfApi
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    try:
        model_id = os.getenv("MODEL_ID")
        cache_dir = os.getenv("HF_HOME")
        token = os.getenv("HF_TOKEN")
        
        if not token:
            logger.error("HF_TOKEN environment variable is required")
            return False

        logger.info(f"Starting download of {model_id}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Get list of files to download
        api = HfApi()
        files = api.list_repo_files(model_id, token=token)
        logger.info(f"Found {len(files)} files to download")
        
        # Download files
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=cache_dir,
            token=token,
            resume_download=True,
            max_workers=1,  # Reduced for stability
            tqdm_class=tqdm
        )
        
        # Verify key files
        key_files = [
            "model_index.json",
            "v3-5-large/diffusion_pytorch_model.safetensors",
            "v3-5-large/config.json"
        ]
        
        missing_files = []
        for file in key_files:
            full_path = os.path.join(local_dir, file)
            if not os.path.exists(full_path):
                missing_files.append(file)
                logger.error(f"Missing file: {file}")
        
        if missing_files:
            logger.error("Download incomplete - missing required files")
            return False
            
        logger.info("Download and verification complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
EOL

# Verify HF_TOKEN is set
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

# Create cache directory with proper permissions
mkdir -p $CACHE_DIR
chmod -R 777 $CACHE_DIR

echo "Starting fresh download..."
echo "Model ID: $MODEL_ID"
echo "Cache directory: $(pwd)/$CACHE_DIR"

# Run the download in a container
docker run --rm \
    -v $(pwd)/$CACHE_DIR:/root/.cache/huggingface \
    -v $(pwd)/download_script.py:/app/download_script.py \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODEL_ID=$MODEL_ID \
    -e HF_TOKEN=${HF_TOKEN} \
    --memory=4g \
    --memory-swap=8g \
    python:3.10-slim \
    bash -c "
        cd /app && \
        echo 'Installing requirements...' && \
        pip install --no-cache-dir --quiet huggingface-hub tqdm && \
        echo 'Starting download...' && \
        python3 download_script.py
    "

# Cleanup the temporary script
rm download_script.py

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download successful! Verifying files..."
    TOTAL_SIZE=$(du -sh "$CACHE_DIR/hub/models--stabilityai--stable-diffusion-3.5-large" | cut -f1)
    echo "Total size: $TOTAL_SIZE"
else
    echo "Error: Download failed"
    exit 1
fi