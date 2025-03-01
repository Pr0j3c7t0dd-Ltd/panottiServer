#!/bin/bash

# ANSI color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Global variables
HAS_SUFFICIENT_GPU_BUFFER=true

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

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
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

# Setup plugin configurations
setup_plugin_configs() {
    print_step "Setting up plugin configurations"
    
    # Find and copy all plugin.yaml.example files to plugin.yaml
    find "$INSTALL_DIR/app/plugins" -name "plugin.yaml.example" | while read -r example_file; do
        target_file="${example_file%.example}"
        if [ ! -f "$target_file" ]; then
            cp "$example_file" "$target_file"
            rel_path="${target_file#$INSTALL_DIR/}"
            print_success "Created ${rel_path}"
        fi
    done
}

# Check system requirements using Swift
check_system_requirements() {
    print_step "Checking system requirements"
    
    # Create temporary Swift script
    cat > /tmp/CheckMaxBufferLength.swift << 'EOL'
import Metal

if let device = MTLCreateSystemDefaultDevice() {
    let maxBufferGB = Double(device.maxBufferLength) / (1024.0 * 1024.0 * 1024.0)
    print(String(format: "%.2f", maxBufferGB))
} else {
    print("0")
}
EOL

    # Run Swift script - this will trigger Xcode Command Line Tools installation if needed
    print_step "Checking GPU capabilities"
    echo "Note: If Swift is not installed, you may be prompted to install Xcode Command Line Tools"
    
    local max_buffer_gb=$(swift /tmp/CheckMaxBufferLength.swift)
    if [ $? -ne 0 ]; then
        print_warning "Swift command failed. Waiting for potential Xcode Command Line Tools installation..."
        sleep 30  # Give time for the installation prompt and potential quick installation
        max_buffer_gb=$(swift /tmp/CheckMaxBufferLength.swift)
        if [ $? -ne 0 ]; then
            print_error "Failed to run Swift check. Please ensure Xcode Command Line Tools are installed and try again."
        fi
    fi
    
    rm /tmp/CheckMaxBufferLength.swift
    
    if (( $(echo "$max_buffer_gb < 8.5" | bc -l) )); then
        HAS_SUFFICIENT_GPU_BUFFER=false
        print_warning "Your system's GPU buffer (${max_buffer_gb}GB) is insufficient for local meeting notes generation.\nMinimum requirement is 8.5GB."
        print_info "However, you can still proceed with installation and use remote meeting notes generation via OpenAI, Anthropic, or Google.\nFor remote meeting notes setup details, visit: https://panotti.io/docs/server after you setup the server."
        print_warning "PRIVACY NOTICE: When using remote meeting notes, your meeting data will be transmitted to the LLM provider (OpenAI, Anthropic, or Google). While we ensure secure transmission, please be aware that this data leaves your local machine and should be considered when handling sensitive information."
        
        read -p "Would you like to proceed with installation? (y/n): " proceed
        if [[ $proceed != "y" && $proceed != "Y" ]]; then
            print_error "Installation cancelled by user."
            exit 1
        fi
        
        # Display warning about API charges
        echo -e "\n⚠️  WARNING: Using remote meeting notes will incur API charges ⚠️"
        echo "You will be responsible for all API charges associated with using this feature."
        echo "These charges will be billed directly by the provider you select."

        # Get user confirmation
        while true; do
            read -p "Do you understand and accept responsibility for all API charges? (y/n) " yn
            case $yn in
                [Yy]* ) break;;
                [Nn]* ) echo "Setup cannot continue without accepting the charges."; exit 1;;
                * ) echo "Please answer yes or no.";;
            esac
        done

    fi
    
    print_success "System requirements met - GPU buffer: ${max_buffer_gb}GB"
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
echo -e "${BLUE}     Welcome to Panotti Server Installation (v1.8)     ${NC}"
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
echo "1. Check your system's requirements"
echo "2. Install Homebrew (package manager)"
echo "3. Install Ollama (for local meeting notes generation)"
echo "4. Install Docker Desktop (for running the server)"
echo "5. Clone and set up the Panotti Server"

echo -e "\n${YELLOW}Note: You may be prompted for your password during installation${NC}"

if ! confirm "Would you like to proceed with the installation?"; then
    echo "Installation cancelled."
    exit 0
fi

# Check system requirements
check_system_requirements

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
if [ "$HAS_SUFFICIENT_GPU_BUFFER" = true ]; then
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
        
        # Check if Ollama is running through Homebrew services
        if brew services list | grep ollama >/dev/null 2>&1; then
            # Ollama is managed by Homebrew
            if ! brew services list | grep ollama | grep started >/dev/null; then
                print_step "Starting Ollama service via Homebrew"
                brew services restart ollama
                
                # Verify service started successfully
                if brew services list | grep ollama | grep started >/dev/null; then
                    print_success "Ollama service started successfully"
                else
                    print_error "Failed to start Ollama service via Homebrew"
                fi
            fi
        else
            # Ollama is installed via official installer
            print_step "Starting Ollama service via systemctl"
            if ! pgrep -x "ollama" >/dev/null; then
                # Try starting Ollama using the official method
                if [ -f "/Applications/Ollama.app/Contents/MacOS/ollama" ]; then
                    open -a Ollama
                    
                    # Wait for Ollama to start
                    COUNTER=0
                    while ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; do
                        if [ $COUNTER -ge 30 ]; then
                            print_error "Failed to start Ollama service. Please start Ollama manually and try again."
                        fi
                        echo "Waiting for Ollama to start... ($COUNTER seconds)"
                        sleep 2
                        COUNTER=$((COUNTER + 2))
                    done
                    print_success "Ollama service started successfully"
                else
                    print_error "Could not find Ollama application. Please start Ollama manually and try again."
                fi
            else
                print_success "Ollama service is already running"
            fi
        fi
    fi
else
    print_info "Skipping Ollama installation as system will use remote processing"
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
cat > "$INSTALL_DIR/.env" << EOL
API_KEY=${API_KEY}
HOST_RECORDINGS_DIR=${RECORDINGS_DIR}
RECORDINGS_DIR=${RECORDINGS_DIR}
EOL

# Configure meeting notes based on GPU check
if [ "$HAS_SUFFICIENT_GPU_BUFFER" = false ]; then
    
    # Setup remote meeting notes
    print_step "Setting up remote meeting notes"

    echo "Please select your preferred meeting notes model provider:"
    echo "1) OpenAI (GPT-4o)"
    echo "2) Anthropic (Claude 3.5 Sonnet)"
    echo "3) Google (Gemini 1.5 Flash)"
    
    while true; do
        read -p "Enter your choice (1-3): " provider_choice
        case $provider_choice in
            1)
                provider="openai"
                print_info "To obtain an OpenAI API key:"
                echo "1. Go to https://platform.openai.com/account/api-keys"
                echo "2. Sign up or log in to your OpenAI account"
                echo "3. Click on 'Create new secret key'"
                echo "4. Copy the generated API key"
                break
                ;;
            2)
                provider="anthropic"
                print_info "To obtain an Anthropic API key:"
                echo "1. Go to https://console.anthropic.com/account/keys"
                echo "2. Sign up or log in to your Anthropic account"
                echo "3. Click on 'Create Key'"
                echo "4. Copy the generated API key"
                break
                ;;
            3)
                provider="google"
                print_info "To obtain a Google API key:"
                echo "1. Go to https://makersuite.google.com/app/apikey"
                echo "2. Sign up or log in to your Google Cloud account"
                echo "3. Click on 'Create API Key'"
                echo "4. Copy the generated API key"
                break
                ;;
            *)
                echo "Invalid choice. Please enter 1, 2, or 3."
                ;;
        esac
    done
    
    # Get API key
    while true; do
        read -p "Please enter your $provider API key: " api_key
        if [[ -n "$api_key" ]]; then
            break
        else
            echo "API key cannot be empty. Please try again."
        fi
    done

    setup_plugin_configs

    # Disable local meeting notes plugin
    if [ -f "$INSTALL_DIR/app/plugins/meeting_notes_local/plugin.yaml" ]; then
        sed -i '' 's/enabled: true/enabled: false/' "$INSTALL_DIR/app/plugins/meeting_notes_local/plugin.yaml"
        print_success "Disabled local meeting notes plugin"
    fi
    
    # Update remote meeting notes plugin configuration
    if [ -f "$INSTALL_DIR/app/plugins/meeting_notes_remote/plugin.yaml" ]; then
        # Enable plugin and set provider
        sed -i '' "s/enabled: .*/enabled: true/" "$INSTALL_DIR/app/plugins/meeting_notes_remote/plugin.yaml"
        sed -i '' "s/^  provider: .*/  provider: $provider/" "$INSTALL_DIR/app/plugins/meeting_notes_remote/plugin.yaml"
        
        # Update API key for the selected provider
        case $provider in
            "openai")
                sed -i '' "/^  openai:/,/^  [a-z]/{s/^    api_key: .*/    api_key: $api_key/}" "$INSTALL_DIR/app/plugins/meeting_notes_remote/plugin.yaml"
                ;;
            "anthropic")
                sed -i '' "/^  anthropic:/,/^  [a-z]/{s/^    api_key: .*/    api_key: $api_key/}" "$INSTALL_DIR/app/plugins/meeting_notes_remote/plugin.yaml"
                ;;
            "google")
                sed -i '' "/^  google:/,/^  [a-z]/{s/^    api_key: .*/    api_key: $api_key/}" "$INSTALL_DIR/app/plugins/meeting_notes_remote/plugin.yaml"
                ;;
        esac
        print_success "Remote meeting notes plugin configured successfully"
    fi
fi

# Run the Docker setup script
print_step "Running Docker setup"
if [ "$HAS_SUFFICIENT_GPU_BUFFER" = false ]; then
    python3 "$INSTALL_DIR/scripts/docker_setup_no_ollama.py" --unattended --api-key="${API_KEY}" --recordings-dir="${RECORDINGS_DIR}"
else
    python3 "$INSTALL_DIR/scripts/docker_setup.py" --unattended --api-key="${API_KEY}" --recordings-dir="${RECORDINGS_DIR}"
fi

print_success "Installation complete!"
echo -e "Installation directory: ${INSTALL_DIR}\n" 
echo -e "Type 'exit' to quit this terminal\n" 