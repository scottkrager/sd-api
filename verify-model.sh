#!/bin/bash

# Set variables
CACHE_DIR=".cache/huggingface"

# Create Python verification script
cat > verify_script.py << 'EOL'
import os
from huggingface_hub import HfApi
import logging
from pathlib import Path
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_files_recursive(base_path, pattern):
    """Recursively find files matching the pattern"""
    return glob.glob(str(Path(base_path) / '**' / pattern), recursive=True)

def verify_model():
    try:
        model_id = "stabilityai/stable-diffusion-3.5-large"
        cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        token = os.getenv("HF_TOKEN")
        
        if not token:
            logger.error("HF_TOKEN environment variable is required")
            return False

        # Check if cache directory exists
        if not os.path.exists(cache_dir):
            logger.error(f"Cache directory {cache_dir} does not exist")
            return False
            
        # List all subdirectories for debugging
        logger.info("Cache directory contents:")
        for root, dirs, files in os.walk(cache_dir):
            level = root.replace(cache_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            if level < 2:  # Only show files for top few levels
                for f in files:
                    logger.info(f"{indent}    {f}")

        # Key files to look for (using partial paths)
        key_files = [
            "model_index.json",
            "diffusion_pytorch_model.safetensors",
            "config.json"
        ]
        
        missing_files = []
        for file_pattern in key_files:
            matches = find_files_recursive(cache_dir, file_pattern)
            if not matches:
                missing_files.append(file_pattern)
            else:
                logger.info(f"Found {file_pattern} at: {matches[0]}")
        
        if missing_files:
            logger.error("Missing required files:")
            for file in missing_files:
                logger.error(f"  - {file}")
            return False
            
        # Check total size of cache
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(cache_dir)
            for filename in filenames
        )
        logger.info(f"Total cache size: {total_size / (1024*1024*1024):.2f} GB")
            
        logger.info("All required model files found!")
        return True
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    success = verify_model()
    if success:
        logger.info("Model verification successful!")
        logger.info("Your model files appear to be complete and ready to use.")
        sys.exit(0)
    else:
        logger.error("Model verification failed!")
        logger.error("Please ensure all required files are downloaded.")
        sys.exit(1)
EOL

# Verify HF_TOKEN is set
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

echo "Starting model verification..."
echo "Cache directory: $(pwd)/$CACHE_DIR"

# Run the verification in a container
docker run --rm \
    -v $(pwd)/$CACHE_DIR:/root/.cache/huggingface \
    -v $(pwd)/verify_script.py:/app/verify_script.py \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN=${HF_TOKEN} \
    python:3.10-slim \
    bash -c "
        cd /app && \
        echo 'Installing requirements...' && \
        pip install --no-cache-dir --quiet huggingface-hub && \
        echo 'Starting verification...' && \
        python3 verify_script.py
    "

VERIFY_EXIT_CODE=$?

# Cleanup the temporary script
rm verify_script.py

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ Verification completed successfully!"
    echo "Cache size: $(du -sh $CACHE_DIR)"
else
    echo "❌ Verification failed!"
    echo "Please check the logs above for details."
fi

exit $VERIFY_EXIT_CODE