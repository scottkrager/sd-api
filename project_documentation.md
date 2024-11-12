# Project Documentation
Generated on Mon Nov 11 17:02:52 PST 2024

## Project Structure
```
.
├── app
│   ├── __init__.py
│   ├── app.py
│   └── monitoring.py
├── cache-manager.sh
├── config
│   ├── grafana
│   │   └── provisioning
│   │       ├── dashboards
│   │       │   └── default.yaml
│   │       └── datasources
│   │           └── default.yaml
│   └── prometheus
│       └── prometheus.yml
├── docker
│   ├── Dockerfile.dev
│   ├── Dockerfile.prod
│   └── nginx
│       └── default.conf
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── document_project.sh
├── download-model.sh
├── download_model.py
├── logs
│   └── nginx
├── project_documentation.md
├── requirements.txt
└── setup.sh

12 directories, 18 files
```

## File Contents

### ./setup.sh
```sh
#!/bin/bash

# Create directory structure
mkdir -p docker/nginx
mkdir -p logs/nginx
mkdir -p config/prometheus
mkdir -p config/grafana/provisioning
mkdir -p generated_images

# Create Prometheus config
cat > config/prometheus/prometheus.yml << 'EOL'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sd-api'
    static_configs:
      - targets: ['sd-api:5000']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOL

# Set permissions
chmod -R 755 config
chmod -R 777 generated_images
chmod -R 777 logs

# Create .env file
cat > .env << 'EOL'
HF_TOKEN=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
MODEL_ID=stabilityai/stable-diffusion-3.5-large
NUM_WORKERS=1
PYTORCH_ENABLE_MPS_FALLBACK=1
LOG_LEVEL=debug
EOL

echo "Setup complete! You can now run: docker-compose -f docker-compose.dev.yml up --build -d"```

### ./docker/Dockerfile.prod
```prod
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p generated_images

ENV MODEL_ID="runwayml/stable-diffusion-v1-5" \
    NUM_WORKERS=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1", "--threads", "4", "--log-level", "debug", "app:app"]```

### ./docker/Dockerfile.dev
```dev
# Use Python base image with CUDA support for production
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /src/app

# Create directory for generated images
RUN mkdir -p generated_images

# Set environment variables
ENV MODEL_ID="stabilityai/stable-diffusion-3.5-large" \
    NUM_WORKERS=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/src

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "--workers", "1", "--threads", "4", "--log-level", "debug", "app.app:app"]```

### ./docker/nginx/default.conf
```conf
server {
    listen 127.0.0.1:80;
    server_name sd-api.test www.sd-api.test *.sd-api.test;
    return 301 https://$host$request_uri;
}

server {
    listen 127.0.0.1:443 ssl http2;
    server_name sd-api.test www.sd-api.test *.sd-api.test;
    root /;
    charset utf-8;
    client_max_body_size 512M;
    
    ssl_certificate "/Users/scottkrager/.config/valet/Certificates/sd-api.test.crt";
    ssl_certificate_key "/Users/scottkrager/.config/valet/Certificates/sd-api.test.key";
    
    location / {
        proxy_pass http://127.0.0.1:8088;  # Updated to match new nginx port
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8088/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 10;
    }
    
    access_log off;
    error_log "/Users/scottkrager/.config/valet/Log/nginx-error.log";
    
    location ~ /\.ht {
        deny all;
    }
}```

### ./app/monitoring.py
```py
import time
import threading
from datetime import datetime
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import requests
import os

# Prometheus metrics
REQUESTS = Counter('sd_api_requests_total', 'Total requests processed', ['endpoint'])
GENERATION_TIME = Histogram('sd_api_generation_seconds', 'Time spent generating images')
GPU_MEMORY = Gauge('sd_api_gpu_memory_bytes', 'GPU memory usage in bytes')
COST_TRACKER = Gauge('sd_api_cost_dollars', 'Estimated cost in dollars')
IDLE_TIME = Gauge('sd_api_idle_seconds', 'Time since last request')

class MonitoringService:
    def __init__(self):
        self.last_request_time = time.time()
        self.start_time = time.time()
        self.runpod_api_key = os.getenv('RUNPOD_API_KEY')
        self.cost_alert_threshold = float(os.getenv('COST_ALERT_THRESHOLD', 10))
        self.max_idle_time = int(os.getenv('MAX_IDLE_TIME', 3600))
        self.current_cost = 0
        
        # Start Prometheus metrics server
        start_http_server(9090)
        
        # Start monitoring threads
        threading.Thread(target=self._monitor_gpu, daemon=True).start()
        threading.Thread(target=self._monitor_idle, daemon=True).start()
        threading.Thread(target=self._monitor_cost, daemon=True).start()

    def record_request(self, endpoint):
        """Record an API request"""
        REQUESTS.labels(endpoint=endpoint).inc()
        self.last_request_time = time.time()
        IDLE_TIME.set(0)

    def record_generation_time(self, duration):
        """Record image generation time"""
        GENERATION_TIME.observe(duration)

    def _monitor_gpu(self):
        """Monitor GPU memory usage"""
        import torch
        while True:
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated()
                GPU_MEMORY.set(memory_used)
            time.sleep(15)

    def _monitor_idle(self):
        """Monitor idle time and shutdown if exceeded"""
        while True:
            idle_time = time.time() - self.last_request_time
            IDLE_TIME.set(idle_time)
            
            # Auto-shutdown if idle for too long
            if idle_time > self.max_idle_time:
                self._shutdown_pod()
            
            time.sleep(60)

    def _monitor_cost(self):
        """Monitor and track costs"""
        # Cost per hour for different GPU types (example rates)
        HOURLY_RATES = {
            'A4000': 0.39,
            'A5000': 0.49,
            'A6000': 0.79
        }
        
        # Get GPU type from environment or default to A4000
        gpu_type = os.getenv('GPU_TYPE', 'A4000')
        hourly_rate = HOURLY_RATES.get(gpu_type, 0.39)
        
        while True:
            # Calculate cost based on time running
            running_time = (time.time() - self.start_time) / 3600  # Convert to hours
            self.current_cost = running_time * hourly_rate
            COST_TRACKER.set(self.current_cost)
            
            # Check cost threshold
            if self.current_cost > self.cost_alert_threshold:
                self._send_cost_alert()
            
            time.sleep(300)  # Update every 5 minutes

    def _send_cost_alert(self):
        """Send cost alert (customize this based on your needs)"""
        # Example: Send to Discord webhook
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if webhook_url:
            alert_data = {
                "content": f"⚠️ Cost Alert: Your Stable Diffusion API has exceeded ${self.cost_alert_threshold:.2f}"
            }
            requests.post(webhook_url, json=alert_data)

    def _shutdown_pod(self):
        """Shutdown the Runpod instance"""
        if self.runpod_api_key:
            try:
                # Get pod ID from hostname
                pod_id = os.uname().nodename
                
                # Call Runpod API to stop pod
                headers = {'Authorization': f'Bearer {self.runpod_api_key}'}
                url = f'https://api.runpod.io/v2/pod/{pod_id}/stop'
                response = requests.post(url, headers=headers)
                
                if response.status_code == 200:
                    print(f"Pod {pod_id} shutdown initiated")
                else:
                    print(f"Failed to shutdown pod: {response.text}")
            
            except Exception as e:
                print(f"Error during shutdown: {str(e)}")
        
        # Fallback to system shutdown
        os.system("shutdown now")
```

### ./app/__init__.py
```py
```

### ./app/app.py
```py
import os
from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import torch
from PIL import Image
import io
import uuid
import time
from app.monitoring import MonitoringService

app = Flask(__name__)

# Initialize the monitoring service
monitor = MonitoringService()

def get_device_and_dtype():
    if torch.backends.mps.is_available() and not os.getenv('FORCE_CPU', False):
        return "mps", torch.float16
    elif torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32

# Initialize the model globally
device, dtype = get_device_and_dtype()
model_id = os.getenv('MODEL_ID', "stabilityai/stable-diffusion-3.5-large")
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Load the Stable Diffusion model with appropriate settings"""
    try:
        app.logger.info(f"Loading model on device: {device} with dtype: {dtype}")
        
        if device == "cuda":
            # Use 4-bit quantization for CUDA devices to reduce VRAM usage
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16
            )
            
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                transformer=model_nf4,
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
        else:
            # For MPS (M1/M2 Mac) or CPU
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_auth_token=os.getenv('HF_TOKEN')
            )
            pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
            
        return pipe
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        raise

# Load the pipeline
try:
    pipe = load_model()
    app.logger.info(f"Model loaded successfully on device: {device}")
except Exception as e:
    app.logger.error(f"Failed to load model: {str(e)}")
    pipe = None

@app.before_request
def before_request():
    """Record request timing"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Record request metrics"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        if request.endpoint:
            monitor.record_request(request.endpoint)
            if request.endpoint == 'generate_image':
                monitor.record_generation_time(duration)
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status"""
    request_info = {
        'headers': dict(request.headers),
        'remote_addr': request.remote_addr,
        'path': request.path,
        'method': request.method
    }
    
    memory_info = {}
    if device == "cuda" and torch.cuda.is_available():
        memory_info = {
            'gpu_memory_allocated': torch.cuda.memory_allocated(),
            'gpu_memory_reserved': torch.cuda.memory_reserved()
        }
    
    return jsonify({
        'status': 'healthy' if pipe is not None else 'degraded',
        'model': model_id,
        'device': device,
        'memory_info': memory_info,
        'request_info': request_info,
        'uptime_seconds': time.time() - monitor.start_time
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    if pipe is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Optional parameters with defaults for SD 3.5
        negative_prompt = data.get('negative_prompt', None)
        num_inference_steps = int(data.get('num_inference_steps', 28))  # SD 3.5 default
        guidance_scale = float(data.get('guidance_scale', 3.5))  # SD 3.5 default
        width = int(data.get('width', 1024))  # SD 3.5 supports higher resolutions
        height = int(data.get('height', 1024))
        max_sequence_length = int(data.get('max_sequence_length', 512))  # SD 3.5 supports longer prompts
        
        app.logger.info(f"Generating image for prompt: {prompt}")
        generation_start = time.time()
        
        # Generate the image
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                max_sequence_length=max_sequence_length
            ).images[0]
        
        generation_time = time.time() - generation_start
        app.logger.info(f"Image generated in {generation_time:.2f} seconds")
        monitor.record_generation_time(generation_time)
        
        # Save the image
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        
        # Prepare response
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(
            img_byte_arr,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        app.logger.error(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)```

### ./config/grafana/provisioning/datasources/default.yaml
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

### ./config/grafana/provisioning/dashboards/default.yaml
```yaml
# grafana/provisioning/dashboards/default.yaml
apiVersion: 1

providers:
  - name: 'Stable Diffusion API'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/dashboards

# grafana/provisioning/datasources/default.yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true```

### ./config/prometheus/prometheus.yml
```yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sd-api'
    static_configs:
      - targets: ['sd-api:5000']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### ./document_project.sh
```sh
#!/bin/bash

# Output file
OUTPUT_FILE="project_documentation.md"

# Create or clear the output file
echo "# Project Documentation" > "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add project structure using tree
echo "## Project Structure" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
tree -I "node_modules|.git|.cache|__pycache__|*.pyc|generated_images|*.log" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add file contents
echo "## File Contents" >> "$OUTPUT_FILE"

# Find all relevant files, excluding certain directories and file types
find . -type f \
    ! -path "*/\.*" \
    ! -path "*/node_modules/*" \
    ! -path "*/__pycache__/*" \
    ! -path "*/generated_images/*" \
    ! -name "*.pyc" \
    ! -name "*.log" \
    ! -name "*.md" \
    ! -name "*.txt" \
    -print0 | while IFS= read -r -d '' file; do
    
    # Get file extension
    extension="${file##*.}"
    
    # Add file header
    echo -e "\n### $file" >> "$OUTPUT_FILE"
    echo '```'"$extension" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
done

echo "Documentation generated in $OUTPUT_FILE"```

### ./docker-compose.dev.yml
```yml
version: '3.8'

services:
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "8088:80"
    volumes:
      - ./docker/nginx:/etc/nginx/conf.d
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - sd-api
    networks:
      - sd-network

  sd-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    expose:
      - "5000"
    volumes:
      - ./app:/src/app
      - ./generated_images:/src/generated_images
      - model_cache:/root/.cache/huggingface  # Add this volume for model caching
    environment:
      - MODEL_ID=${MODEL_ID}
      - NUM_WORKERS=1
      - PYTORCH_ENABLE_MPS_FALLBACK=1
      - LOG_LEVEL=debug
      - PYTHONPATH=/src
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
      - HF_HOME=/root/.cache/huggingface
    networks:
      - sd-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.45.0
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9091:9090"
    networks:
      - sd-network

  grafana:
    image: grafana/grafana:10.0.0
    volumes:
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3001:3000"
    networks:
      - sd-network

networks:
  sd-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
  model_cache:  # Add this volume definition```

### ./download-model.sh
```sh
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
fi```

### ./download_model.py
```py
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
MODEL_ID = "runwayml/stable-diffusion-v1-5"
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
    exit(download_model())```

### ./cache-manager.sh
```sh
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
esac```

### ./docker-compose.prod.yml
```yml
version: '3.8'

services:
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "8080:80"
    volumes:
      - ./docker/nginx:/etc/nginx/conf.d
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - sd-api
    networks:
      - sd-network

  sd-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    expose:
      - "5000"
    volumes:
      - ./generated_images:/app/generated_images
    environment:
      - RUNPOD_API_KEY=${RUNPOD_API_KEY}
      - GPU_TYPE=A4000  # Or your chosen GPU type
      - COST_ALERT_THRESHOLD=10
      - MAX_IDLE_TIME=3600
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - sd-network
    restart: unless-stopped

  # ... rest of services remain the same as dev ...

networks:
  sd-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:```
