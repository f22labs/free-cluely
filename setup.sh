#!/bin/bash

# Setup script for Free Cluely
# This script installs all dependencies and sets up the project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "Starting Free Cluely setup..."
echo ""

# Check prerequisites
print_info "Checking prerequisites..."

# Check Node.js
if ! command_exists node; then
    print_error "Node.js is not installed. Please install Node.js from https://nodejs.org/"
    exit 1
fi
NODE_VERSION=$(node --version)
print_success "Node.js found: $NODE_VERSION"

# Check npm
if ! command_exists npm; then
    print_error "npm is not installed. Please install npm."
    exit 1
fi
NPM_VERSION=$(npm --version)
print_success "npm found: $NPM_VERSION"

# Check Python 3
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3."
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_success "Python 3 found: $PYTHON_VERSION"

# Check pip
if ! command_exists pip3 && ! command_exists pip; then
    print_warning "pip not found. Attempting to install..."
    if command_exists python3; then
        python3 -m ensurepip --upgrade || print_warning "Could not install pip automatically. Please install pip manually."
    fi
fi

if command_exists pip3; then
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_CMD="pip"
else
    print_error "pip is not available. Please install pip."
    exit 1
fi
PIP_VERSION=$($PIP_CMD --version)
print_success "pip found: $PIP_VERSION"

echo ""

# Install Node.js dependencies
print_info "Installing Node.js dependencies..."
print_warning "This may take a few minutes..."

# Clean install approach for Sharp compatibility
if [ -d "node_modules" ]; then
    print_info "Removing existing node_modules..."
    rm -rf node_modules
fi

if [ -f "package-lock.json" ]; then
    print_info "Removing existing package-lock.json..."
    rm -f package-lock.json
fi

# Install with Sharp workaround
print_info "Installing dependencies (with Sharp workaround)..."
SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install --ignore-scripts

print_info "Rebuilding Sharp..."
npm rebuild sharp

print_success "Node.js dependencies installed!"

# Install renderer dependencies if package.json exists
if [ -f "renderer/package.json" ]; then
    print_info "Installing renderer dependencies..."
    cd renderer
    if [ -d "node_modules" ]; then
        rm -rf node_modules
    fi
    npm install
    cd ..
    print_success "Renderer dependencies installed!"
fi

echo ""

# Install Python dependencies
print_info "Installing Python dependencies..."
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

$PIP_CMD install -r requirements.txt
print_success "Python dependencies installed!"

echo ""

# Setup .env file
print_info "Setting up environment variables..."
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating template..."
    cat > .env << 'EOF'
# AI Provider Configuration
# Choose ONE of the following options:

# Option 1: Google Gemini (Cloud AI)
# GEMINI_API_KEY=your_api_key_here
# Get your API key from: https://makersuite.google.com/app/apikey

# Option 2: Ollama (Local/Private AI - Recommended for Privacy)
USE_OLLAMA=true
OLLAMA_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
# Make sure Ollama is installed and running: https://ollama.ai
EOF
    print_success ".env file created!"
    print_warning "Please edit .env file and configure your AI provider settings."
else
    print_success ".env file already exists. Skipping creation."
fi

echo ""

# Make Python scripts executable
print_info "Making Python scripts executable..."
chmod +x realtime_stt_service.py 2>/dev/null || true
chmod +x mic_testing.py 2>/dev/null || true
chmod +x test_realtime_stt.py 2>/dev/null || true
print_success "Python scripts are now executable!"

echo ""

# Final summary
print_success "Setup completed successfully!"
echo ""
print_info "Next steps:"
echo "  1. Edit .env file to configure your AI provider (Gemini or Ollama)"
echo "  2. Test your microphone: python3 mic_testing.py"
echo "  3. Start the app: npm start"
echo ""
print_info "For more information, see README.md"
echo ""