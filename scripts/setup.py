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
    """Check if Python version is 3.12.x and offer pyenv installation if needed"""
    if sys.version_info.major != 3 or sys.version_info.minor != 12:
        print(f"Current Python version: {platform.python_version()}")
        print("Python 3.12.x is required.")
        
        # Check if we're on macOS first
        if platform.system() != "Darwin":
            print("You'll need to install Python 3.12.x manually on your system.")
            sys.exit(1)
            
        if get_user_confirmation("Would you like to install Python 3.12 using pyenv?"):
            try:
                # Check if pyenv is installed
                subprocess.run(["pyenv", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("pyenv is not installed. Installing via Homebrew...")
                try:
                    subprocess.run(["brew", "install", "pyenv"], check=True)
                except subprocess.CalledProcessError as e:
                    print("Error installing pyenv. Please ensure Homebrew is properly installed.")
                    sys.exit(1)

            # Install Python 3.12 using pyenv
            print("Installing Python 3.12 via pyenv...")
            subprocess.run(["pyenv", "install", "3.12"], check=True)
            print("Setting local Python version to 3.12...")
            subprocess.run(["pyenv", "local", "3.12"], check=True)
            print("Python 3.12 installed and configured successfully. Please restart your terminal and run this script again.")
            sys.exit(0)
        else:
            print("Please install Python 3.12.x manually and run this script again.")
            sys.exit(1)


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
        activate_script = ".venv/bin/activate" if platform.system() != "Windows" else ".venv\\Scripts\\activate"
        subprocess.run([f"source {activate_script} && poetry install"], shell=True, check=True)
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
        print("Downloading Whisper model...")
        script_path = Path(
            "app/plugins/audio_transcription_local/scripts/download_models.py"
        )
        if script_path.exists():
            subprocess.run(
                [sys.executable, str(script_path), "--model", "base.en"], check=True
            )
    else:
        print("Whisper model is required for audio transcription functionality.")
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
        print("\nLocal meeting note processing requires Ollama (https://ollama.com/download)")
        if get_user_confirmation("Have you already installed Ollama on your machine?"):
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
            print(f"Node.js version {node_version} is installed, but version 18 or higher is required.")
            if platform.system() == "Darwin" and get_user_confirmation("Would you like to install Node.js 18 via Homebrew?"):
                subprocess.run(["brew", "install", "node@18"], check=True)
                # Link the installed version
                subprocess.run(["brew", "link", "node@18", "--force"], check=True)
                print("Node.js 18 installed successfully!")
            else:
                print("Please install Node.js 18 or higher manually from https://nodejs.org/")
                sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if platform.system() == "Darwin" and get_user_confirmation("Node.js is not installed. Would you like to install it via Homebrew?"):
            subprocess.run(["brew", "install", "node@18"], check=True)
            subprocess.run(["brew", "link", "node@18", "--force"], check=True)
            print("Node.js installed successfully!")
        else:
            print("Please install Node.js 18 or higher manually from https://nodejs.org/")
            sys.exit(1)


def setup_admin_frontend():
    """Setup the admin frontend application"""
    print("\nSetting up admin frontend...")
    admin_dir = Path("admin-frontend")
    
    if not admin_dir.exists():
        print("Error: admin-frontend directory not found!")
        sys.exit(1)
        
    # Copy .env.local if it doesn't exist
    env_example = admin_dir / ".env.local.sample"
    env_target = admin_dir / ".env.local"
    if not env_target.exists() and env_example.exists():
        shutil.copy(env_example, env_target)
        print("Created admin frontend .env.local from sample file")
    
    # Install npm dependencies
    os.chdir(admin_dir)
    print("\nInstalling admin frontend dependencies...")
    subprocess.run(["npm", "install"], check=True)
    
    # Run password initialization
    print("\nInitializing admin password...")
    subprocess.run(["npm", "run", "init-password"], check=True)
    
    os.chdir("..")
    print("Admin frontend setup completed successfully!")


def main():
    """Main setup function"""
    try:
        print("\nWelcome to the panottiServer setup script!")
        print("This script will guide you through the installation process.")
        print("You can choose which components to install.\n")
        
        print("Important Note:")
        print("If you plan to customize the code or create your own plugins,")
        print("please first fork the repository before running this setup script:")
        print("https://github.com/Pr0j3c7t0dd-Ltd/panottiServer\n")

        if not get_user_confirmation("Would you like to proceed with the setup?"):
            print("Setup cancelled.")
            sys.exit(0)

        # Check Ollama setup first
        check_ollama_setup()

        # First check and install Homebrew as it's needed for other dependencies
        check_brew_installation()
        # Then install system dependencies including pyenv
        install_system_dependencies()
        # Now check Python version since we have the tools to install it if needed
        check_python_version()
        
        check_rust_installation()
        check_poetry_installation()
        setup_virtual_environment()
        # check_docker_installation()
        download_whisper_model()
        create_ssl_directory()
        check_node_installation()
        setup_admin_frontend()
        copy_env_file()
        copy_plugin_yaml_files()

        print("\nSetup completed successfully!")
        print("\nImportant Next Steps:")
        print("1. Review and configure your plugin settings:")
        print("   - Check app/plugins/meeting_notes/plugin.yaml")
        print("   - By default, local meeting note processing is enabled")
        print("   - To use remote processing, enable the remote_meeting_notes plugin")
        print("   - Add your API keys in the remote_meeting_notes plugin configuration")
        print("\n2. Start the server using one of the following commands:")
        print("   a. Using the shell script (recommended):")
        print("      ./start_servers.sh")
        print("   b. Using Docker Compose:")
        print("      docker-compose up")
        print("\n3. After you start the server, you can access the admin frontend:")
        print("   - Visit http://localhost:54790/")
        print("   - Default password: Pa55w0rd")
        print("   - You will be prompted to change this password on first login")
        print("\nMake sure all configuration files are properly set up before starting the server.")

    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
