#!/bin/bash

# Script to manage model cache

CACHE_DIR="$(pwd)/.model_cache"
DOCKER_VOLUME_NAME="model_cache"

download_model() {
    echo "Downloading model to cache directory..."
    docker run --rm \
        -v ${CACHE_DIR}:/model_cache \
        -e HF_HOME=/model_cache \
        -e TRANSFORMERS_CACHE=/model_cache \
        python:3.10-slim \
        bash -c "pip install --no-cache-dir torch diffusers transformers accelerate safetensors && \
                python3 -c 'from diffusers import StableDiffusionPipeline; \
                import torch; \
                model_id = \"runwayml/stable-diffusion-v1-5\"; \
                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)'"
}

case "$1" in
    "download")
        download_model
        ;;
    "clear")
        echo "Clearing model cache..."
        rm -rf ${CACHE_DIR}/*
        docker volume rm ${DOCKER_VOLUME_NAME} 2>/dev/null || true
        ;;
    "status")
        echo "Cache directory size:"
        du -sh ${CACHE_DIR}
        echo "Docker volume info:"
        docker volume inspect ${DOCKER_VOLUME_NAME}
        ;;
    *)
        echo "Usage: $0 {download|clear|status}"
        exit 1
        ;;
esac