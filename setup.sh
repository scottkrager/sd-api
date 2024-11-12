#!/bin/bash

# Create directory structure
mkdir -p docker/nginx
mkdir -p logs/nginx
mkdir -p generated_images
mkdir -p config/prometheus
mkdir -p config/grafana/{dashboards,provisioning/{dashboards,datasources,plugins,notifiers,alerting}}

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

# Create Grafana dashboard provisioning config
cat > config/grafana/provisioning/dashboards/default.yaml << 'EOL'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    folderUid: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/dashboards
      foldersFromFilesStructure: true
EOL

# Create Grafana datasource config
cat > config/grafana/provisioning/datasources/default.yaml << 'EOL'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOL

# Create default dashboard
cat > config/grafana/dashboards/sd-api-dashboard.json << 'EOL'
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "PBFA97CFB590B2093"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true
      },
      "title": "Total Requests",
      "type": "gauge",
      "targets": [
        {
          "expr": "sd_api_requests_total",
          "refId": "A"
        }
      ]
    }
  ],
  "refresh": "5s",
  "schemaVersion": 38,
  "style": "dark",
  "tags": ["stable-diffusion"],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Stable Diffusion API Dashboard",
  "uid": "stable-diffusion",
  "version": 1
}
EOL

# Create placeholder files for other Grafana directories
touch config/grafana/provisioning/plugins/.gitkeep
touch config/grafana/provisioning/notifiers/.gitkeep
touch config/grafana/provisioning/alerting/.gitkeep

# Set permissions
chmod -R 755 config
chmod -R 777 generated_images
chmod -R 777 logs
chmod -R 777 config/grafana

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << 'EOL'
HF_TOKEN=your_huggingface_token_here
MODEL_ID=stabilityai/stable-diffusion-3.5-large
NUM_WORKERS=1
PYTORCH_ENABLE_MPS_FALLBACK=1
LOG_LEVEL=debug
EOL
    echo "Created .env file. Please update HF_TOKEN with your Hugging Face token."
fi

# Create empty directories for Grafana SQLite database
mkdir -p data/grafana
chmod -R 777 data/grafana

echo "Setup complete! You can now run: docker compose -f docker-compose.dev.yml up --build"