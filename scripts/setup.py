#!/usr/bin/env python3
"""
Setup script for panottiServer application.
Handles environment setup, dependencies, and initial configuration.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import platform


def check_python_version():
    """Check if Python version is 3.12.x"""
    if sys.version_info.major != 3 or sys.version_info.minor != 12:
        print("Error: Python 3.12.x is required")
        print(f"Current Python version: {platform.python_version()}")
        sys.exit(1)


def check_rust_installation():
    """Check if Rust is installed"""
    try:
        subprocess.run(['rustc', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Rust is not installed. Installing Rust...")
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            subprocess.run('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh', shell=True, check=True)
            # Source the environment
            os.environ['PATH'] = f"{str(Path.home())}/.cargo/bin:{os.environ['PATH']}"
        else:
            print("Please install Rust manually from https://rustup.rs/")
            sys.exit(1)


def check_poetry_installation():
    """Check if Poetry is installed, install if not"""
    try:
        subprocess.run(['poetry', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Poetry is not installed. Installing Poetry...")
        subprocess.run('curl -sSL https://install.python-poetry.org | python3 -', shell=True, check=True)


def setup_virtual_environment():
    """Setup virtual environment using Poetry"""
    print("Setting up virtual environment and installing dependencies...")
    subprocess.run(['poetry', 'install'], check=True)


def copy_env_file():
    """Copy .env.example to .env if it doesn't exist"""
    if not os.path.exists('.env'):
        shutil.copy('.env.example', '.env')
        print("Created .env file from .env.example")


def copy_plugin_yaml_files():
    """Copy plugin.yaml.example files to plugin.yaml for each plugin"""
    plugins_dir = Path('app/plugins')
    for plugin_dir in plugins_dir.iterdir():
        if plugin_dir.is_dir() and not plugin_dir.name.startswith('__'):
            example_yaml = plugin_dir / 'plugin.yaml.example'
            target_yaml = plugin_dir / 'plugin.yaml'
            if example_yaml.exists() and not target_yaml.exists():
                shutil.copy(example_yaml, target_yaml)
                print(f"Created {target_yaml} from example file")


def download_whisper_model():
    """Download the Whisper model"""
    print("Downloading Whisper model...")
    script_path = Path('app/plugins/audio_transcription_local/scripts/download_models.py')
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path), '--model', 'base.en'], check=True)


def create_ssl_directory():
    """Create SSL directory and generate self-signed certificates"""
    ssl_dir = Path('ssl')
    if not ssl_dir.exists():
        ssl_dir.mkdir()
        os.chdir(ssl_dir)
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-nodes',
            '-out', 'cert.pem', '-keyout', 'key.pem', '-days', '365',
            '-subj', '/CN=localhost'
        ], check=True)
        os.chdir('..')
        print("Created SSL certificates")


def main():
    """Main setup function"""
    try:
        print("Starting panottiServer setup...")
        
        # Check requirements
        check_python_version()
        check_rust_installation()
        check_poetry_installation()
        
        # Setup steps
        setup_virtual_environment()
        copy_env_file()
        copy_plugin_yaml_files()
        download_whisper_model()
        create_ssl_directory()
        
        print("\nSetup completed successfully!")
        print("\nTo start the server, run one of the following commands:")
        print("1. Using Python directly:")
        print("   poetry run python run_server.py")
        print("\n2. Using the shell script:")
        print("   ./start_server.sh")
        print("\nThe server will be available at:")
        print("- HTTP:  http://localhost:8001")
        print("- HTTPS: https://localhost:8001")
        print("\nAPI documentation will be available at:")
        print("- Swagger UI: http://localhost:8001/docs")
        print("- ReDoc:     http://localhost:8001/redoc")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 