#!/usr/bin/env python3
"""
Setup script for panottiServer application.
Handles environment setup, dependencies, and initial configuration.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ANSI Color codes
class Colors:
    HEADER = '\x1b[38;5;26m'     # Blue
    BLUE = '\x1b[38;5;26m'       # Blue
    GREEN = '\033[92m'      # Green
    YELLOW = '\033[93m'     # Yellow
    RED = '\033[91m'        # Red
    BOLD = '\033[1m'        # Bold
    UNDERLINE = '\033[4m'   # Underline
    END = '\033[0m'         # Reset

def color_text(text, color):
    """Wrap text with color codes"""
    return f"{color}{text}{Colors.END}"

def print_step(emoji, text):
    """Print a setup step with consistent formatting"""
    print("\n" + emoji + " " + color_text(text, Colors.BLUE))

def print_success(text):
    """Print a success message"""
    print(color_text(text, Colors.GREEN))

def print_warning(text):
    """Print a warning message"""
    print(color_text(text, Colors.YELLOW))

def print_error(text):
    """Print an error message"""
    print(color_text(text, Colors.RED))

def get_user_confirmation(message):
    """Ask user for confirmation before proceeding with an action"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please enter 'y' for yes or 'n' for no.")


def get_user_input(prompt, default=None):
    """Get user input with an optional default value"""
    if default:
        response = input(f"{prompt} (default: {default}): ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()


def update_env_value(file_path, key, value):
    """Update a specific key's value in the .env file"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip() and not line.startswith('#'):
                if line.split('=')[0].strip() == key:
                    file.write(f"{key}={value}\n")
                else:
                    file.write(line)
            else:
                file.write(line)


def check_python_version():
    """Check if Python version is 3.12.x and handle pyenv version switching"""
    
    # First ensure Homebrew is installed
    if not check_brew_installation():
        print_error("Homebrew is required for pyenv installation.")
        sys.exit(1)

    # Check/Install pyenv
    try:
        subprocess.run(["pyenv", "--version"], check=True, capture_output=True)
        print("pyenv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing pyenv via Homebrew...")
        try:
            subprocess.run(["brew", "install", "pyenv"], check=True)
        except subprocess.CalledProcessError as e:
            print_error("Error installing pyenv")
            sys.exit(1)

    # Update shell configuration if needed
    shell_rc = os.path.expanduser("~/.zshrc" if os.environ.get("SHELL", "").endswith("zsh") else "~/.bashrc")
    with open(shell_rc, "r") as f:
        rc_content = f.read()
    
    pyenv_init = '''
# pyenv initialization
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
'''
    if "pyenv init" not in rc_content:
        with open(shell_rc, "a") as f:
            f.write(pyenv_init)
        print("Added pyenv initialization to shell configuration")
        print_warning("Please run the following command and then run this script again:")
        print(f"source {shell_rc}")
        sys.exit(0)

    # Set up environment for current session
    pyenv_root = os.path.expanduser("~/.pyenv")
    os.environ["PYENV_ROOT"] = pyenv_root
    os.environ["PATH"] = f"{pyenv_root}/bin:{os.environ['PATH']}"

    # Check if Python 3.12 is installed in pyenv
    try:
        versions_output = subprocess.run(["pyenv", "versions"], capture_output=True, text=True, check=True).stdout
        if "3.12" not in versions_output:
            print("Installing Python 3.12 via pyenv...")
            subprocess.run(["pyenv", "install", "3.12"], check=True)
    except subprocess.CalledProcessError as e:
        print_error("Failed to check/install Python 3.12")
        print_error("Please run these commands manually and try again:")
        print("eval \"$(pyenv init --path)\"")
        print("eval \"$(pyenv init -)\"")
        print("pyenv install 3.12")
        sys.exit(1)

    # Set local version to 3.12
    try:
        subprocess.run(["pyenv", "local", "3.12"], check=True)
    except subprocess.CalledProcessError:
        print_error("Failed to set Python 3.12 as local version")
        sys.exit(1)

    # Verify we're using the correct version
    try:
        version_check = subprocess.run(["python", "--version"], capture_output=True, text=True, check=True)
        if "3.12" not in version_check.stdout:
            print_warning("Python 3.12 is not active. Please restart your shell and run this script again.")
            sys.exit(0)
        print_success("Python 3.12 is properly configured and active.")
    except subprocess.CalledProcessError:
        print_warning("Could not verify Python version. Please restart your shell and run this script again.")
        sys.exit(0)


def check_brew_installation():
    """Check if Homebrew is installed on macOS"""
    if platform.system() != "Darwin":
        print("Not on macOS, skipping Homebrew checks")
        return False

    try:
        subprocess.run(["brew", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        if get_user_confirmation("Homebrew is not installed. Would you like to install it?"):
            subprocess.run(
                '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
                shell=True,
                check=True,
            )
            return True
        else:
            print("Homebrew is required for system dependencies. Please install it manually.")
            return False


def install_system_dependencies():
    """Install required system dependencies"""
    if not check_brew_installation():
        return

    print("The following system dependencies are required:")
    brew_packages = [
        ("terminal-notifier", "Required for desktop notifications"),
        ("ffmpeg", "Required for audio processing"),
        ("pyenv", "Recommended for Python version management")
    ]

    for package, description in brew_packages:
        try:
            # Check if package is already installed
            result = subprocess.run(
                ["brew", "list", package], capture_output=True, check=False
            )
            if result.returncode == 0:
                print(f"{package} is already installed ({description})")
            else:
                if get_user_confirmation(f"Would you like to install {package}? ({description})"):
                    print(f"Installing {package}...")
                    subprocess.run(["brew", "install", package], check=True)
                else:
                    print(f"Skipping {package} installation...")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            if not get_user_confirmation("Would you like to continue with the setup?"):
                sys.exit(1)


def check_rust_installation():
    """Check if Rust is installed"""
    try:
        subprocess.run(["rustc", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if get_user_confirmation("Rust is not installed. Would you like to install it?"):
            if platform.system() == "Darwin" or platform.system() == "Linux":
                subprocess.run(
                    'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh',
                    shell=True,
                    check=True,
                )
                # Source the environment
                os.environ["PATH"] = f"{Path.home()!s}/.cargo/bin:{os.environ['PATH']}"
            else:
                print("Please install Rust manually from https://rustup.rs/")
                sys.exit(1)
        else:
            print("Rust is required for some dependencies. Please install it manually.")
            sys.exit(1)


def check_poetry_installation():
    """Check if Poetry is installed, install if not"""
    try:
        # First try to check if poetry is in PATH
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("Poetry is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If not in PATH, check if it's installed via Homebrew
        try:
            subprocess.run(["brew", "list", "poetry"], check=True, capture_output=True)
            print("Poetry is installed via Homebrew but not in PATH")
            if get_user_confirmation("Would you like to add Poetry to your PATH?"):
                # Add Poetry to PATH
                subprocess.run(["brew", "link", "poetry", "--force"], check=True)
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            if get_user_confirmation("Poetry is not installed. Would you like to install it using Homebrew?"):
                subprocess.run(["brew", "install", "poetry"], check=True)
                return True
            else:
                print("Poetry is required for dependency management. Please install it manually.")
                sys.exit(1)
    return False


def setup_virtual_environment():
    """Setup virtual environment using venv"""
    if get_user_confirmation("Would you like to set up the virtual environment and install dependencies?"):
        print("Setting up virtual environment using venv...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        
        venv_python = ".venv/bin/python"
        venv_pip = ".venv/bin/pip"

        # Upgrade pip in the virtual environment
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install poetry using the venv's pip
        subprocess.run([venv_pip, "install", "poetry"], check=True)
        
        # Use the venv's Python to run poetry install
        subprocess.run([venv_python, "-m", "poetry", "install"], check=True)
    else:
        print("Virtual environment setup skipped. Note that this is required for the application to run.")
        sys.exit(1)


def copy_env_file():
    """Copy .env.example to .env if it doesn't exist and configure key values"""
    if not os.path.exists(".env"):
        if get_user_confirmation("Would you like to create a .env file from .env.example?"):
            shutil.copy(".env.example", ".env")
            print("Created .env file from .env.example")
            
            print("\nNow let's configure some important settings:")
            
            # Get API_KEY
            print("\nThe API_KEY should match the one set in your Panotti desktop app.")
            api_key = get_user_input("Enter your API_KEY", "your_api_key_here")
            update_env_value(".env", "API_KEY", api_key)
            
            # Update admin frontend .env.local
            admin_env_path = Path("admin-frontend/.env.local")
            if admin_env_path.exists():
                update_env_value(admin_env_path, "NEXT_PUBLIC_API_KEY", api_key)
                print("Updated admin frontend API key configuration")
            
            # Get RECORDINGS_DIR
            print("\nThe RECORDINGS_DIR should point to the same recordings directory set in your Panotti desktop app.")
            recordings_dir = get_user_input("Enter the path to your recordings directory")
            # Ensure the path is properly quoted
            recordings_dir = f'"{recordings_dir}"'
            update_env_value(".env", "RECORDINGS_DIR", recordings_dir)
            
            print("\nEnvironment file configured successfully!")
        else:
            print("Environment file is required for the application to run.")
            sys.exit(1)


def copy_plugin_yaml_files():
    """Copy plugin.yaml.example files to plugin.yaml for each plugin"""
    if get_user_confirmation("Would you like to set up all plugin configuration files with default settings?"):
        plugins_dir = Path("app/plugins")
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith("__"):
                example_yaml = plugin_dir / "plugin.yaml.example"
                target_yaml = plugin_dir / "plugin.yaml"
                if example_yaml.exists() and not target_yaml.exists():
                    shutil.copy(example_yaml, target_yaml)
                    print(f"Created {target_yaml} from example file")
        print("All plugin configurations have been set up with default settings")
    else:
        print("Plugin configuration is required for the application to run.")
        sys.exit(1)


def download_whisper_model():
    """Download the Whisper model"""
    if get_user_confirmation("Would you like to download the Whisper model? This is required for audio transcription."):
        print_step("🎙️", "Downloading Whisper model (this may take a few minutes)...")
        script_path = Path("app/plugins/audio_transcription_local/scripts/download_models.py")
        if script_path.exists():
            try:
                # Use the virtual environment's Python if available
                python_exec = os.path.join(".venv", "bin", "python") if os.path.exists(".venv") else sys.executable
                subprocess.run([
                    python_exec,
                    str(script_path),
                    "--model",
                    "base.en"
                ], check=True, capture_output=True, text=True)
                print_success("Whisper model downloaded successfully!")
            except subprocess.CalledProcessError as e:
                print_error(f"Error downloading Whisper model: {e.stderr}")
                if not get_user_confirmation("Would you like to continue with setup?"):
                    sys.exit(1)
        else:
            print_error("Download script not found!")
            if not get_user_confirmation("Would you like to continue with setup?"):
                sys.exit(1)
    else:
        print_warning("Whisper model is required for audio transcription functionality.")
        sys.exit(1)


def create_ssl_directory():
    """Create SSL directory and generate self-signed certificates"""
    if get_user_confirmation("Would you like to create SSL certificates for HTTPS support?"):
        ssl_dir = Path("ssl")
        if not ssl_dir.exists():
            ssl_dir.mkdir()
            os.chdir(ssl_dir)
            print("Generating self-signed SSL certificates...")
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:4096",
                    "-nodes",
                    "-out",
                    "cert.pem",
                    "-keyout",
                    "key.pem",
                    "-days",
                    "365",
                    "-subj",
                    "/CN=localhost",
                ],
                check=True,
            )
            print("SSL certificates generated successfully")
            os.chdir("..")
    else:
        print("SSL certificates are required for secure HTTPS connections.")
        sys.exit(1)


def check_docker_installation():
    """Check if Docker is installed and configured correctly"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if get_user_confirmation("Docker or Docker Compose is not installed. Would you like to install them?"):
            if platform.system() == "Darwin":
                print("Please install Docker Desktop from https://www.docker.com/products/docker-desktop")
            else:
                print("Please install Docker and Docker Compose manually.")
            sys.exit(1)
        else:
            print("Docker is required for running the application.")
            sys.exit(1)


def check_ollama_setup():
    """Check if user wants to use local meeting note processing and setup Ollama"""
    if get_user_confirmation("\nDo you plan to process meeting notes locally on your machine?"):
        print("\n" + color_text('⚠️  Important Note:', Colors.YELLOW))
        print("Local meeting note processing requires Ollama (https://ollama.com/download)")
        if get_user_confirmation("\nHave you already installed Ollama on your machine?"):
            try:
                # Check if ollama is available
                subprocess.run(["ollama", "--version"], check=True, capture_output=True)
                
                # Ask about default model
                print("\nThe default model for local processing is 'llama3.1:8b'")
                print("Note: You can use any other Ollama model, but you'll need to update")
                print("      the model name in the plugin configuration files.")
                if get_user_confirmation("Would you like to pull the default model now?"):
                    print("\nPulling llama3.1:8b model (this may take a while)...")
                    subprocess.run(["ollama", "pull", "llama3.1:8b"], check=True)
                    print("Model downloaded successfully!")
                else:
                    print("\nSkipping model download.")
                    print("Remember to update the model name in app/plugins/meeting_notes/plugin.yaml")
                    print("if you plan to use a different model.")
            except subprocess.CalledProcessError:
                print("\nError: Ollama is not properly installed or not in PATH")
                print("Please install Ollama from https://ollama.com/download")
                if not get_user_confirmation("Would you like to continue with setup?"):
                    sys.exit(1)
        else:
            print("\nPlease install Ollama from https://ollama.com/download")
            print("You can continue with setup and install Ollama later.")
            if not get_user_confirmation("Would you like to continue with setup?"):
                sys.exit(1)
    else:
        print("\nSkipping Ollama setup. You'll need to configure remote processing")
        print("in app/plugins/remote_meeting_notes/plugin.yaml")


def check_node_installation():
    """Check if Node.js is installed and at the correct version"""
    try:
        # Check Node.js version
        node_version = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True).stdout.strip()
        version_num = node_version.lstrip('v').split('.')
        major_version = int(version_num[0])
        
        if major_version < 18:
            print_warning(f"Node.js version {node_version} is installed, but version 18 or higher is required.")
            if platform.system() == "Darwin" and get_user_confirmation("Would you like to install Node.js 18 via Homebrew?"):
                subprocess.run(["brew", "install", "node@18"], check=True)
                # Link the installed version
                subprocess.run(["brew", "link", "node@18", "--force"], check=True)
                print_success("Node.js 18 installed successfully!")
            else:
                print_warning("Please install Node.js 18 or higher manually from https://nodejs.org/")
                sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if platform.system() == "Darwin" and get_user_confirmation("Node.js is not installed. Would you like to install it via Homebrew?"):
            subprocess.run(["brew", "install", "node@18"], check=True)
            subprocess.run(["brew", "link", "node@18", "--force"], check=True)
            print_success("Node.js installed successfully!")
        else:
            print_warning("Please install Node.js 18 or higher manually from https://nodejs.org/")
            sys.exit(1)


def setup_admin_frontend():
    """Setup the admin frontend application"""
    admin_dir = Path("admin-frontend")
    
    if not admin_dir.exists():
        print_error("Error: admin-frontend directory not found!")
        sys.exit(1)
        
    # Copy .env.local if it doesn't exist
    env_example = admin_dir / ".env.local.sample"
    env_target = admin_dir / ".env.local"
    if not env_target.exists() and env_example.exists():
        shutil.copy(env_example, env_target)
        print_success("Created admin frontend .env.local from sample file")
    
    # Install npm dependencies
    os.chdir(admin_dir)
    print("\nInstalling admin frontend dependencies...")
    subprocess.run(["npm", "install"], check=True)
    
    # Run password initialization
    print("\nInitializing admin password...")
    subprocess.run(["npm", "run", "init-password"], check=True)
    
    os.chdir("..")
    print_success("Admin frontend setup completed successfully!")


def main():
    """Main setup function"""
    try:
        print("\n" + color_text('🚀 Welcome to the panottiServer setup script!', Colors.HEADER))
        print("This script will guide you through the installation process.")
        print("You can choose which components to install.\n")
        
        print("\n" + color_text('⚠️  Important Note:', Colors.YELLOW))
        print("If you plan to customize the code or create your own plugins,")
        print("please first fork the repository before running this setup script:")
        print(color_text("https://github.com/Pr0j3c7t0dd-Ltd/panottiServer\n", Colors.UNDERLINE))

        if not get_user_confirmation("Would you like to proceed with the setup?"):
            print_warning("Setup cancelled.")
            sys.exit(0)

        # Check Ollama setup first
        print_step("🤖", "Checking Ollama setup...")
        check_ollama_setup()

        print_step("🍺", "Checking Homebrew installation...")
        check_brew_installation()
        
        print_step("📦", "Installing system dependencies...")
        install_system_dependencies()
        
        print_step("🐍", "Checking Python version...")
        check_python_version()
        
        print_step("💿", "Checking Rust installation...")
        check_rust_installation()
        
        print_step("📝", "Checking Poetry installation...")
        check_poetry_installation()
        
        print_step("🌐", "Setting up virtual environment...")
        setup_virtual_environment()
        
        print_step("🎙️", "Downloading Whisper model...")
        download_whisper_model()
        
        print_step("🔒", "Setting up SSL certificates...")
        create_ssl_directory()
        
        print_step("💻", "Checking Node.js installation...")
        check_node_installation()
        
        print_step("🖥️", "Setting up admin frontend...")
        setup_admin_frontend()
        
        print_step("⚡", "Setting up environment files...")
        copy_env_file()
        
        print_step("🔧", "Setting up plugin configurations...")
        copy_plugin_yaml_files()

        print("\n" + color_text('✨ Setup completed successfully! 💥', Colors.GREEN + Colors.BOLD))
        print("\n" + color_text('📋 Important Next Steps:', Colors.HEADER))
        print(color_text("1. Review and configure your plugin settings:", Colors.BOLD))
        print("   - Check app/plugins/meeting_notes/plugin.yaml")
        print("   - By default, local meeting note processing is enabled")
        print("   - To use remote processing, enable the remote_meeting_notes plugin")
        print("   - Add your API keys in the remote_meeting_notes plugin configuration")
        print("\n" + color_text('2. Start the server using one of the following commands:', Colors.BOLD))
        print("   a. Using the shell script (recommended):")
        print(color_text("\n     👉 ./start_servers.sh 👈\n", Colors.GREEN))
        print("   b. Using Docker Compose:")
        print(color_text("      docker-compose up", Colors.GREEN))
        print("\n" + color_text('3. After you start the server, you can access the admin frontend:', Colors.BOLD))
        print(color_text("   - 👉 Visit http://localhost:54790/  👈", Colors.BLUE))
        print("   - Default password: Pa55w0rd")
        print("   - You will be prompted to change this password on first login")
        print("\n" + color_text('🎯 Make sure all configuration files are properly set up before starting the server.', Colors.YELLOW))

    except KeyboardInterrupt:
        print_error("\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ An error occurred during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
