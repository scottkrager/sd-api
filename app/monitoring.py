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
