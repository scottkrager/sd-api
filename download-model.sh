#!/bin/bash

# Set variables
CACHE_DIR=".cache/huggingface"
MODEL_ID=${MODEL_ID:-"stabilityai/stable-diffusion-3.5-large"}

# Create Python download script
cat > download_script.py << 'EOL'
import os
import time
import sys
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
import logging
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_size(model_id, token):
    try:
        api = HfApi()
        model_info = api.model_info(model_id, token=token)
        total_size = sum(s.size for s in model_info.siblings if s.size is not None)
        return total_size
    except Exception as e:
        logger.warning(f"Couldn't get model size: {e}")
        return None

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def download_model():
    try:
        model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3.5-large")
        cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        token = os.getenv("HF_TOKEN")
        
        if not token:
            logger.error("HF_TOKEN environment variable is required")
            return False

        # Get model size before downloading
        total_size = get_model_size(model_id, token)
        if total_size:
            logger.info(f"Total model size: {format_size(total_size)}")

        logger.info(f"Starting download of {model_id}")
        logger.info(f"Cache directory: {cache_dir}")
        
        start_time = time.time()
        
        # Download with progress bar and resume capability
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=cache_dir,
            token=token,
            resume_download=True,
            max_workers=4,
            tqdm_class=tqdm,
            force_download=False
        )
        
        duration = time.time() - start_time
        
        # Calculate downloaded size
        total_downloaded = sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, files in os.walk(local_dir)
            for name in files
        )
        
        logger.info(f"Download completed in {duration:.1f} seconds")
        logger.info(f"Average speed: {format_size(total_downloaded/duration)}/s")
        logger.info(f"Total downloaded: {format_size(total_downloaded)}")
        return True
        
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted. You can resume later.")
        return False
    except HfHubHTTPError as e:
        if "401" in str(e):
            logger.error("Authentication error. Please check your HF_TOKEN.")
        elif "404" in str(e):
            logger.error(f"Model {model_id} not found. Please check the model ID.")
        else:
            logger.error(f"Download failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
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

echo "Starting model download process..."
echo "Model ID: $MODEL_ID"
echo "Cache directory: $(pwd)/$CACHE_DIR"

# Run the download in a container with resource limits
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
    echo "Model download completed successfully!"
    echo "Cache location: $(pwd)/$CACHE_DIR"
    
    # Verify the download
    echo "Verifying downloaded files..."
    FILE_COUNT=$(find $CACHE_DIR -type f | wc -l)
    TOTAL_SIZE=$(du -sh $CACHE_DIR | cut -f1)
    echo "Found $FILE_COUNT files in cache directory"
    echo "Total size: $TOTAL_SIZE"
    
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