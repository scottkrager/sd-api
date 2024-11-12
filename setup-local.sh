#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up SD-API for local development...${NC}\n"

# 1. Ensure .env file exists and has required values
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << 'EOL'
HF_TOKEN=your_huggingface_token_here
MODEL_ID=stabilityai/stable-diffusion-3.5-large
NUM_WORKERS=1
PYTORCH_ENABLE_MPS_FALLBACK=1
LOG_LEVEL=debug
EOL
    echo "Created .env file - please update HF_TOKEN with your token"
fi

# 2. Create required directories
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p logs/nginx logs/sd-api generated_images
chmod -R 777 logs generated_images

# 3. Set up Valet configuration
echo -e "${GREEN}Setting up Valet configuration...${NC}"
if command -v valet &> /dev/null; then
    # Backup existing configuration if it exists
    if [ -f "$HOME/.config/valet/Nginx/sd-api.test" ]; then
        mv "$HOME/.config/valet/Nginx/sd-api.test" "$HOME/.config/valet/Nginx/sd-api.test.backup"
    fi
    
    # Copy our Valet configuration
    cp resources/valet.conf "$HOME/.config/valet/Nginx/sd-api.test"
    
    # Restart Valet
    echo -e "${YELLOW}Restarting Valet...${NC}"
    valet restart
else
    echo -e "${YELLOW}Warning: Valet not found. Please install Valet first.${NC}"
    exit 1
fi

# 4. Start Docker services
echo -e "${GREEN}Starting Docker services...${NC}"
docker compose -f docker-compose.dev.yml up -d

# 5. Add hosts entry if not present
if ! grep -q "sd-api.test" /etc/hosts; then
    echo -e "${YELLOW}Adding hosts entry...${NC}"
    echo "127.0.0.1 sd-api.test" | sudo tee -a /etc/hosts
fi

# 6. Verify services are running
echo -e "${GREEN}Verifying services...${NC}"
echo "Waiting for services to start..."
sleep 5

# Check if services are running
if docker compose -f docker-compose.dev.yml ps | grep -q "running"; then
    echo -e "${GREEN}Services are running!${NC}"
    echo -e "\nYou can now access the API at: https://sd-api.test"
    echo -e "To test the API, try:"
    echo -e "curl -X POST https://sd-api.test/generate \\"
    echo -e "  -H 'Content-Type: application/json' \\"
    echo -e "  -d '{\"prompt\": \"a beautiful mountain landscape\"}'"
else
    echo -e "${YELLOW}Warning: Services may not have started properly. Check logs with:${NC}"
    echo "docker compose -f docker-compose.dev.yml logs"
fi

# Print status
echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "To view logs, run: docker compose -f docker-compose.dev.yml logs -f"
echo -e "To stop services, run: docker compose -f docker-compose.dev.yml down"