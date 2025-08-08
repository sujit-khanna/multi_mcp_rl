#!/bin/bash
# Automated Setup Script for Multi-MCP RL Training
# Designed for CUDA-enabled Linux machines

set -e  # Exit on any error

echo "🚀 Starting Multi-MCP RL Training Setup..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems. Detected: $OSTYPE"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. CUDA functionality may not be available."
else
    print_status "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi

# Step 1: Install system dependencies
print_step "Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential libnuma-dev git git-lfs curl python3.12 python3.12-venv python3.12-dev

# Step 2: Install uv if not present
if ! command -v uv &> /dev/null; then
    print_step "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
else
    print_status "uv is already installed: $(uv --version)"
fi

# Step 3: Clone repository if not in it already
if [[ ! -f "requirements.txt" ]]; then
    print_step "Cloning Multi-MCP RL repository..."
    git clone https://github.com/sujit-khanna/multi_mcp_rl.git
    cd multi_mcp_rl
else
    print_status "Already in Multi-MCP RL repository"
fi

# Step 4: Create virtual environment
print_step "Creating Python 3.12 virtual environment..."
uv venv --python 3.12
source .venv/bin/activate

# Step 5: Clone and install SkyRL
print_step "Installing SkyRL framework..."
if [[ ! -d "SkyRL" ]]; then
    git clone https://github.com/novasky-ai/SkyRL.git
fi

# Install SkyRL components
cd SkyRL/skyrl-train
uv pip install -e .
print_status "SkyRL core training installed"

cd ../skyrl-gym
uv pip install -e .
print_status "SkyRL gym installed"

cd ../..

# Step 6: Install project requirements
print_step "Installing project dependencies..."
uv pip install -r requirements.txt

# Step 7: Install CUDA-optimized PyTorch if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    print_step "Installing CUDA-optimized PyTorch..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Step 8: Set up Ray environment
print_step "Configuring Ray environment..."
echo 'export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook' >> ~/.bashrc
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Step 9: Create template .env file
if [[ ! -f ".env" ]]; then
    print_step "Creating template .env file..."
    cat > .env << 'EOF'
# Required API Keys - Replace with your actual keys
OPENAI_API_KEY=your-openai-key-here
POLYGON_API_KEY=your-polygon-api-key
FMP_API_KEY=your-fmp-api-key
TAVILY_API_KEY=tvly-your-tavily-key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token

# WandB Configuration (optional)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=skyrl-tool-training

# Weave Configuration (optional)
WEAVE_PROJECT=your-org/skyrl-tool-training
EOF
    print_warning "Template .env file created. Please update with your actual API keys."
else
    print_status ".env file already exists"
fi

# Step 10: Make MCP servers executable
print_step "Setting up MCP servers..."
chmod +x mcp_tools/limited/*.py

# Step 11: Create directories
mkdir -p outputs data/processed data/inputs

# Step 12: Start Ray (optional)
if command -v ray &> /dev/null; then
    print_step "Starting Ray cluster..."
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 || print_warning "Ray may already be running"
fi

# Step 13: Verification
print_step "Verifying installation..."
python -c "
import torch
import transformers
import peft
import accelerate
try:
    import skyrl_gym
    skyrl_status = '✅ Installed'
except ImportError:
    skyrl_status = '❌ Not found'

print('Installation Verification:')
print(f'  Python: {torch.__version__.split(\"+\")[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  Accelerate: {accelerate.__version__}')
print(f'  SkyRL-Gym: {skyrl_status}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
print(f'  GPU Count: {torch.cuda.device_count()}')
"

echo
print_status "Setup completed successfully! 🎉"
echo
echo "Next steps:"
echo "1. Update .env file with your actual API keys"
echo "2. Ensure training data is in data/processed/ or data/inputs/"
echo "3. Run training: ./training/scripts/launch_real_env_training.sh"
echo
echo "For troubleshooting, see SETUP.md for detailed instructions."
echo
echo "Activate the environment with: source .venv/bin/activate"