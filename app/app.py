import os
from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import uuid
import time

app = Flask(__name__)

def get_device_and_dtype():
    if torch.backends.mps.is_available() and not os.getenv('FORCE_CPU', False):
        return "mps", torch.float16
    elif torch.cuda.is_available():
        return "cuda", torch.float32
    return "cpu", torch.float32

# Initialize the model globally
device, dtype = get_device_and_dtype()
model_id = os.getenv('MODEL_ID', "runwayml/stable-diffusion-v1-5")
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Load the Stable Diffusion model with appropriate settings"""
    try:
        app.logger.info(f"Loading model on device: {device} with dtype: {dtype}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
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
        'request_info': request_info
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
        
        # Optional parameters
        negative_prompt = data.get('negative_prompt', None)
        num_inference_steps = int(data.get('num_inference_steps', 50))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        
        app.logger.info(f"Generating image for prompt: {prompt}")
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
        app.logger.info(f"Image generated in {generation_time:.2f} seconds")
        
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
    app.run(host='0.0.0.0', port=5000)