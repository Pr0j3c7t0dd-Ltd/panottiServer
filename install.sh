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

# Parse command line arguments
API_KEY=""
RECORDINGS_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key=*)
            API_KEY="${1#*=}"
            shift
            ;;
        --recordings-dir=*)
            RECORDINGS_DIR="${1#*=}"
            shift
            ;;
        *)
            print_error "Unknown parameter: $1"
            ;;
    esac
done

# Welcome banner
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}     Welcome to Panotti Server Installation      ${NC}"
echo -e "${BLUE}================================================${NC}"

echo -e "\n${YELLOW}⚠ IMPORTANT: User Responsibility Notice${NC}"
echo -e "It is your responsibility to review and understand the changes this installer will make to your system."
echo -e "Please inspect the installation files and server setup before proceeding to ensure it matches your expectations."
echo -e "This installer will modify system configurations, install software, and create directories.\n"

echo -e "\n${YELLOW}⚠ WARRANTY DISCLAIMER${NC}"
echo -e "THIS SOFTWARE IS PROVIDED \"AS IS\" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE SOFTWARE IS WITH YOU. SHOULD THE SOFTWARE PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR, OR CORRECTION."
echo -e "\nIN NO EVENT SHALL PR0J3CTTODD LTD BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THE SOFTWARE.\n"
echo -e "For complete terms of use and privacy policy, please visit:"
echo -e "- Terms and Conditions: https://www.panotti.io/terms-and-conditions"
echo -e "- Privacy Policy: https://www.panotti.io/privacy-policy\n"

# Installation overview and consent
echo -e "This installation will:"
echo "1. Install Homebrew (package manager)"
echo "2. Install Ollama (for local meeting notes generation)"
echo "3. Install Docker Desktop (for running the server)"
echo "4. Clone and set up the Panotti Server"

echo -e "\n${YELLOW}Note: You may be prompted for your password during installation${NC}"

if ! confirm "Would you like to proceed with the installation?"; then
    echo "Installation cancelled."
    exit 0
fi

# Check if installation directory already exists
INSTALL_DIR="$HOME/panotti-server"
if [ -d "$INSTALL_DIR" ]; then
    print_warning "Directory $INSTALL_DIR already exists"
    if ! confirm "Would you like to remove it and proceed with a fresh installation?"; then
        print_error "Installation cancelled. Please remove or rename the existing directory and try again."
    fi
    rm -rf "$INSTALL_DIR"
    print_success "Existing installation directory removed"
fi

# Get required information upfront
print_step "Required Information"

# Get API Key only if not provided via command line
if [[ -z "$API_KEY" ]]; then
    echo -e "\nThe API_KEY should match the one set in your Panotti desktop app  ('Calllbacks' -> X-API-Key setup for each callback)."
    while true; do
        read -p "Enter your Panotti API Key (default: your_api_key_here): " API_KEY
        API_KEY=${API_KEY:-your_api_key_here}
        if [[ -n "$API_KEY" ]]; then
            break
        else
            echo "API Key cannot be empty. Please try again."
        fi
    done
fi

# Get Recordings Directory only if not provided via command line
if [[ -z "$RECORDINGS_DIR" ]]; then
    echo -e "\nThe recordings directory should match the one configured in your Panotti desktop app ('Settings' -> 'Recordings Location' -> click 'Copy Path' button)."
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
else
    # Create directory if it doesn't exist when provided via command line
    mkdir -p "$RECORDINGS_DIR" 2>/dev/null
    if [[ ! -d "$RECORDINGS_DIR" ]]; then
        print_error "Unable to create or access recordings directory: $RECORDINGS_DIR"
    fi
fi

# Install Homebrew if not present
print_step "Checking for Homebrew installation"
if ! command_exists brew; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for the current session
    eval "$(/opt/homebrew/bin/brew shellenv)"
    
    # Add Homebrew to .zshrc if not already present
    if ! grep -q "eval \"\$(/opt/homebrew/bin/brew shellenv)\"" "$HOME/.zshrc"; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zshrc"
    fi
    
    print_success "Homebrew installed and configured successfully"
else
    # Ensure Homebrew is in PATH for current session
    if ! command -v brew >/dev/null 2>&1; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    # Ensure Homebrew is in .zshrc
    if ! grep -q "eval \"\$(/opt/homebrew/bin/brew shellenv)\"" "$HOME/.zshrc"; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zshrc"
        print_success "Added Homebrew to .zshrc"
    fi
    
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
    
    # Configure Ollama service for startup
    print_step "Configuring Ollama service"
    brew services stop ollama 2>/dev/null  # Stop if running
    brew services start ollama
    
    # Verify service is running
    if brew services list | grep ollama | grep started >/dev/null; then
        print_success "Ollama service started successfully"
    else
        print_error "Failed to start Ollama service"
    fi
    
    # Pull the default model
    print_step "Pulling default Ollama model (llama3.1:8b)"
    ollama pull llama3.1:8b
else
    print_success "Ollama is already installed"
    
    # Ensure service is running and configured for startup
    if ! brew services list | grep ollama | grep started >/dev/null; then
        print_step "Starting Ollama service"
        brew services restart ollama
        
        # Verify service started successfully
        if brew services list | grep ollama | grep started >/dev/null; then
            print_success "Ollama service started successfully"
        else
            print_error "Failed to start Ollama service"
        fi
    fi
fi

# Install Docker if not present
print_step "Checking for Docker installation"
if ! command_exists docker; then
    echo "Installing Docker Desktop..."
    brew install --cask docker
    
    print_success "Docker Desktop installed successfully"
    
    # Start Docker Desktop
    print_step "Starting Docker Desktop"
    open -a Docker
    
    echo "Waiting for Docker Desktop to initialize..."
    echo "Note: You may need to accept the Docker Desktop license agreement if this is your first time."
    echo "Please check for any Docker Desktop windows that may have opened."
    
    # Wait for Docker to be running with a timeout
    TIMEOUT=180  # 3 minutes timeout
    COUNTER=0
    while ! docker info >/dev/null 2>&1; do
        if [ $COUNTER -ge $TIMEOUT ]; then
            print_error "Docker failed to start within ${TIMEOUT} seconds. Please start Docker Desktop manually and try again."
        fi
        echo "Waiting for Docker to start... ($COUNTER seconds)"
        sleep 5
        COUNTER=$((COUNTER + 5))
    done
else
    print_success "Docker Desktop is already installed"
fi

# Ensure Docker is running
print_step "Checking Docker status"
if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Starting Docker Desktop..."
    open -a Docker
    
    # Wait for Docker to be running with a timeout
    TIMEOUT=60  # 1 minute timeout for existing installation
    COUNTER=0
    while ! docker info >/dev/null 2>&1; do
        if [ $COUNTER -ge $TIMEOUT ]; then
            print_error "Docker failed to start within ${TIMEOUT} seconds. Please start Docker Desktop manually and try again."
        fi
        echo "Waiting for Docker to start... ($COUNTER seconds)"
        sleep 5
        COUNTER=$((COUNTER + 5))
    done
fi
print_success "Docker is running"

# Clone the repository
print_step "Cloning Panotti Server repository"
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
echo -e "Installation directory: ${INSTALL_DIR}\n" 
echo -e "Type 'exit' to quit this terminal\n" 