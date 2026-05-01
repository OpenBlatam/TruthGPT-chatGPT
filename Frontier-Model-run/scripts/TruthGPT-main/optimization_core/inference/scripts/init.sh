#!/bin/bash
# 🎯 Initialization Script
# Setup script for inference API environment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Setting up Inference API Environment${NC}\n"

# Check Python
check_python() {
    echo -e "${YELLOW}Checking Python...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 is required but not installed."
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
}

# Create virtual environment
create_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment already exists${NC}"
    fi
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    pip install --upgrade pip
    pip install -r requirements_advanced.txt || pip install fastapi uvicorn httpx pydantic
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Create .env file
create_env() {
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Creating .env file...${NC}"
        cat > .env << EOF
# API Configuration
TRUTHGPT_API_TOKEN=changeme-$(openssl rand -hex 16)
TRUTHGPT_CONFIG=configs/llm_default.yaml
PORT=8080

# Batching
BATCH_MAX_SIZE=32
BATCH_FLUSH_TIMEOUT_MS=20

# Rate Limiting
RATE_LIMIT_RPM=600
RATE_LIMIT_WINDOW_SEC=60

# Cache
CACHE_BACKEND=memory
REDIS_URL=redis://localhost:6379/0

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
ENABLE_STRUCTURED_LOGGING=true

# Webhooks
WEBHOOK_HMAC_SECRET=changeme-secret-$(openssl rand -hex 16)
EOF
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠ Please update TRUTHGPT_API_TOKEN and WEBHOOK_HMAC_SECRET in .env${NC}"
    else
        echo -e "${GREEN}✓ .env file already exists${NC}"
    fi
}

# Create directories
create_directories() {
    echo -e "${YELLOW}Creating directories...${NC}"
    mkdir -p logs
    mkdir -p backups
    mkdir -p data
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Main setup
main() {
    check_python
    create_venv
    install_dependencies
    create_env
    create_directories
    
    echo -e "\n${GREEN}✅ Setup complete!${NC}"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "  1. Review and update .env file"
    echo "  2. Run: source venv/bin/activate"
    echo "  3. Run: python -m uvicorn inference.api:app --reload"
}

main


