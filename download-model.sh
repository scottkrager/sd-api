#!/bin/bash

# Set variables
CACHE_DIR=".cache/huggingface"
MODEL_ID="runwayml/stable-diffusion-v1-5"
DOWNLOAD_SCRIPT=$(cat << 'EOF'
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import logging
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    try:
        # Configure download parameters
        model_id = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
        cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        
        logger.info(f"Starting download of {model_id}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Download with progress bar and resume capability
        snapshot_download(
            repo_id=model_id,
            local_dir=cache_dir,
            resume_download=True,
            max_workers=4,  # Increase number of download workers
            tqdm_class=tqdm
        )
        
        logger.info("Download completed successfully!")
        return True
        
    except HfHubHTTPError as e:
        if "401" in str(e):
            logger.error("Authentication error. Try setting the HF_TOKEN environment variable.")
        else:
            logger.error(f"Download failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        return False

if __name__ == "__main__":
    download_model()
EOF
)

# Create cache directory
mkdir -p $CACHE_DIR

echo "Starting model download process..."
echo "Cache directory: $(pwd)/$CACHE_DIR"

# Run the download in a container
docker run --rm \
    -v $(pwd)/$CACHE_DIR:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODEL_ID=$MODEL_ID \
    -e HF_TOKEN=${HF_TOKEN:-""} \
    python:3.10-slim \
    bash -c "
        echo 'Installing requirements...' && \
        pip install --no-cache-dir --quiet huggingface-hub tqdm && \
        echo 'Starting download...' && \
        python3 -c '$DOWNLOAD_SCRIPT'
    "

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Model download completed successfully!"
    echo "Cache location: $(pwd)/$CACHE_DIR"
    
    # Verify the download
    echo "Verifying downloaded files..."
    FILE_COUNT=$(find $CACHE_DIR -type f | wc -l)
    echo "Found $FILE_COUNT files in cache directory"
    
    if [ $FILE_COUNT -gt 0 ]; then
        echo "Download verification successful!"
    else
        echo "Warning: No files found in cache directory"
        exit 1
    fi
else
    echo "Error: Model download failed"
    exit 1
fi