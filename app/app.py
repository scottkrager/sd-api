import os
from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import io
import uuid
import time
import threading
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = app.logger

# Global variables
pipe = None
model_loading = False
model_load_error = None
model_load_progress = {'status': 'Not started', 'progress': 0}

def get_device_and_dtype():
    if torch.backends.mps.is_available() and not os.getenv('FORCE_CPU', False):
        return "mps", torch.float16
    elif torch.cuda.is_available():
        return "cuda", torch.float32
    return "cpu", torch.float32

def load_model_in_thread():
    global pipe, model_loading, model_load_error, model_load_progress
    try:
        device, dtype = get_device_and_dtype()
        logger.info(f"Loading model on device: {device} with dtype: {dtype}")
        model_loading = True
        model_load_progress['status'] = 'Loading'
        
        # Get model ID and cache dir
        model_id = os.getenv('MODEL_ID', "runwayml/stable-diffusion-v1-5")
        cache_dir = os.getenv('HF_HOME', "/root/.cache/huggingface")
        
        # First try loading with local files
        try:
            logger.info("Attempting to load model from local cache...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_auth_token=os.getenv('HF_TOKEN'),
                safety_checker=None,  # Disable safety checker for memory
                local_files_only=True,
                low_cpu_mem_usage=True,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
                cache_dir=cache_dir
            )
        except Exception as local_error:
            logger.warning(f"Local load failed: {str(local_error)}")
            logger.info("Attempting to download model...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_auth_token=os.getenv('HF_TOKEN'),
                safety_checker=None,  # Disable safety checker for memory
                low_cpu_mem_usage=True,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
                cache_dir=cache_dir
            )
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable optimizations
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing(slice_size="max")
        
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        
        # Use DPM++ 2M scheduler for better quality
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        model_loading = False
        model_load_progress['status'] = 'Ready'
        model_load_progress['progress'] = 100
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        model_load_error = error_msg
        model_loading = False
        model_load_progress['status'] = 'Error'

# Start model loading in background
threading.Thread(target=load_model_in_thread, daemon=True).start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    global pipe, model_loading, model_load_error, model_load_progress
    
    device, _ = get_device_and_dtype()
    
    memory_info = {}
    if device == "cuda" and torch.cuda.is_available():
        memory_info = {
            'gpu_memory_allocated': torch.cuda.memory_allocated(),
            'gpu_memory_reserved': torch.cuda.memory_reserved()
        }
    elif device == "cpu":
        import psutil
        memory = psutil.Process().memory_info()
        memory_info = {
            'rss': memory.rss,
            'vms': memory.vms,
            'shared': memory.shared
        }
    
    status = {
        'status': 'loading' if model_loading else 'error' if model_load_error else 'ready' if pipe else 'not_initialized',
        'model': os.getenv('MODEL_ID', "runwayml/stable-diffusion-v1-5"),
        'device': device,
        'memory_info': memory_info,
        'loading_progress': model_load_progress,
        'error': model_load_error
    }
    
    # Return 503 if still loading or error
    if model_loading:
        return jsonify(status), 503
    elif model_load_error:
        return jsonify(status), 500
    
    return jsonify(status)

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image endpoint"""
    global pipe, model_loading, model_load_error
    
    if model_loading:
        return jsonify({
            'error': 'Model is still loading', 
            'progress': model_load_progress
        }), 503
        
    if model_load_error:
        return jsonify({'error': f'Model failed to load: {model_load_error}'}), 500
        
    if pipe is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        negative_prompt = data.get('negative_prompt', None)
        num_inference_steps = int(data.get('num_inference_steps', 50))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        
        logger.info(f"Generating image for prompt: {prompt}")
        generation_start = time.time()
        
        # Generate the image
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        generation_time = time.time() - generation_start
        logger.info(f"Image generated in {generation_time:.2f} seconds")
        
        # Save the image
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("generated_images", filename)
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
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)