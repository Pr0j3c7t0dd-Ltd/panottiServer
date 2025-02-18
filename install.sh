#!/bin/bash

# ANSI color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Print functions
print_step() {
    echo -e "\n${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Get user confirmation
confirm() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Welcome banner
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}     Welcome to Panotti Server Installation      ${NC}"
echo -e "${BLUE}================================================${NC}"

# Installation overview and consent
echo -e "\nThis installation will:"
echo "1. Install Homebrew (package manager)"
echo "2. Install Ollama (for local meeting notes generation)"
echo "3. Install Docker Desktop (for running the server)"
echo "4. Clone and set up the Panotti Server"

echo -e "\n${YELLOW}Note: You may be prompted for your password during installation${NC}"

if ! confirm "Would you like to proceed with the installation?"; then
    echo "Installation cancelled."
    exit 0
fi

# Get required information upfront
print_step "Required Information"
echo -e "\nThe API_KEY should match the one set in your Panotti desktop app  ('Calllbacks' -> X-API-Key setup for each callback)."

# Get API Key
while true; do
    read -p "Enter your Panotti API Key (default: your_api_key_here): " API_KEY
    API_KEY=${API_KEY:-your_api_key_here}
    if [[ -n "$API_KEY" ]]; then
        break
    else
        echo "API Key cannot be empty. Please try again."
    fi
done

# Get Recordings Directory
echo -e "\nThe recordings directory should match the one configured in your Panotti desktop app ('Settings' -> 'Recordings Location')."
while true; do
    read -p "Enter the full path to your recordings directory: " RECORDINGS_DIR
    if [[ -n "$RECORDINGS_DIR" ]]; then
        # Create directory if it doesn't exist
        mkdir -p "$RECORDINGS_DIR" 2>/dev/null
        if [[ -d "$RECORDINGS_DIR" ]]; then
            break
        else
            echo "Unable to create or access directory. Please check permissions and try again."
        fi
    else
        echo "Directory path cannot be empty. Please try again."
    fi
done

# Install Homebrew if not present
print_step "Checking for Homebrew installation"
if ! command_exists brew; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for the current session
    eval "$(/opt/homebrew/bin/brew shellenv)"
    
    print_success "Homebrew installed successfully"
else
    print_success "Homebrew is already installed"
fi

# Install git if not present
if ! command_exists git; then
    print_step "Installing Git"
    brew install git
    print_success "Git installed successfully"
fi

# Install Ollama if not present
print_step "Checking for Ollama installation"
if ! command_exists ollama; then
    echo "Installing Ollama..."
    brew install ollama
    
    # Start Ollama service
    brew services start ollama
    
    print_success "Ollama installed successfully"
    
    # Pull the default model
    print_step "Pulling default Ollama model (llama3.1:8b)"
    ollama pull llama3.1:8b
else
    print_success "Ollama is already installed"
fi

# Install Docker if not present
print_step "Checking for Docker installation"
if ! command_exists docker; then
    echo "Installing Docker Desktop..."
    brew install --cask docker
    
    print_success "Docker Desktop installed successfully"
    echo "Please start Docker Desktop from your Applications folder"
    echo "After starting Docker Desktop, press any key to continue..."
    read -n 1 -s
else
    print_success "Docker Desktop is already installed"
fi

# Wait for Docker to be running
print_step "Checking Docker status"
while ! docker info >/dev/null 2>&1; do
    echo "Waiting for Docker to start..."
    sleep 5
done
print_success "Docker is running"

# Clone the repository
print_step "Cloning Panotti Server repository"
INSTALL_DIR="$HOME/panotti-server"
if [ -d "$INSTALL_DIR" ]; then
    print_warning "Directory $INSTALL_DIR already exists"
    if ! confirm "Would you like to remove it and clone again?"; then
        print_error "Installation cancelled. Please remove or rename the existing directory and try again."
    fi
    rm -rf "$INSTALL_DIR"
fi

git clone https://github.com/Pr0j3c7t0dd-Ltd/panottiServer.git "$INSTALL_DIR"
cd "$INSTALL_DIR" || print_error "Failed to enter installation directory"
print_success "Repository cloned successfully"

# Create and populate .env file
print_step "Configuring environment"
cat > .env << EOL
API_KEY=${API_KEY}
HOST_RECORDINGS_DIR=${RECORDINGS_DIR}
RECORDINGS_DIR=${RECORDINGS_DIR}
EOL

# Run the Docker setup script
print_step "Running Docker setup"
python3 scripts/docker_setup.py --unattended --api-key="${API_KEY}" --recordings-dir="${RECORDINGS_DIR}"

print_success "Installation complete!"
echo -e "\nYou can access your Panotti Server at: http://localhost:54790"
echo "Default password: Pa55w0rd"
echo -e "API Endpoint: http://localhost:54789\n"
echo -e "Installation directory: ${INSTALL_DIR}\n" 