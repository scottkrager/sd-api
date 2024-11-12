#!/usr/bin/env python3

import os
import time
from huggingface_hub import snapshot_download
import logging
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# MODEL_ID = os.getenv("MODEL_ID", MODEL_ID)
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
CONCURRENT_DOWNLOADS = 8

def download_model():
    try:
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        logger.info(f"Starting download of {MODEL_ID}")
        logger.info(f"Cache directory: {CACHE_DIR}")
        logger.info(f"Using {CONCURRENT_DOWNLOADS} concurrent downloads")
        
        # Start the download with progress bar
        start_time = time.time()
        
        downloaded_snapshot = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=CONCURRENT_DOWNLOADS,
            tqdm_class=tqdm
        )
        
        # Calculate statistics
        duration = time.time() - start_time
        total_size = sum(
            os.path.getsize(os.path.join(downloaded_snapshot, f))
            for f in os.listdir(downloaded_snapshot)
            if os.path.isfile(os.path.join(downloaded_snapshot, f))
        )
        size_mb = total_size / (1024 * 1024)
        
        logger.info(f"Download completed in {duration:.2f} seconds")
        logger.info(f"Total size: {size_mb:.2f}MB")
        logger.info(f"Average speed: {size_mb/duration:.2f}MB/s")
        
        # Verify the download
        logger.info("Verifying downloaded files...")
        files = os.listdir(downloaded_snapshot)
        logger.info(f"Found {len(files)} files in {downloaded_snapshot}")
        logger.info("Files downloaded successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(download_model())