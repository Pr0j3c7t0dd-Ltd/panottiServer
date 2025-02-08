#!/usr/bin/env python3
"""
Quick setup script for running panottiServer in Docker.
Handles configuration files and environment setup for Docker deployment.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ANSI Color codes for pretty output
class Colors:
    BLUE = '\x1b[38;5;26m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def print_step(text):
    """Print a setup step with consistent formatting"""
    print(f"\n{Colors.BLUE}🔧 {text}{Colors.END}")

def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Print an error message and exit"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")
    sys.exit(1)

def get_user_input(prompt, default=None):
    """Get user input with an optional default value"""
    if default:
        response = input(f"{prompt} (default: {default}): ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()

def get_user_confirmation(message):
    """Ask user for confirmation"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please enter 'y' for yes or 'n' for no.")

def check_docker():
    """Check if Docker is installed and running"""
    print_step("Checking Docker installation")

    # First ask if Docker Desktop is installed
    if not get_user_confirmation("Have you installed Docker Desktop?"):
        print_warning("\nDocker Desktop is required to run this application.")
        print("Please download and install Docker Desktop from:")
        print(f"{Colors.BLUE}https://www.docker.com/products/docker-desktop/{Colors.END}")
        print("\nAfter installing Docker Desktop, please run this script again.")
        sys.exit(0)
    
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        # Check if Docker daemon is running
        subprocess.run(["docker", "info"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print_error("Docker is not running. Please start Docker Desktop and try again.")
    except FileNotFoundError:
        print_error("Docker command not found. Please ensure Docker Desktop is properly installed.")

def setup_plugin_configs():
    """Copy plugin.yaml.example files to plugin.yaml for each plugin"""
    print_step("Setting up plugin configurations")
    
    plugin_dir = Path("app/plugins")
    for example_file in plugin_dir.rglob("plugin.yaml.example"):
        target_file = example_file.parent / "plugin.yaml"
        if not target_file.exists():
            shutil.copy2(example_file, target_file)
            print_success(f"Created {target_file.relative_to(Path.cwd())}")

def setup_env_files():
    """Set up .env and admin frontend .env.local files"""
    print_step("Setting up environment files")
    
    # Main .env file
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print_error(".env.example file not found!")
    
    if not env_file.exists():
        shutil.copy2(env_example, env_file)
        print_success("Created .env file")
    
    # Admin frontend .env.local
    admin_env_sample = Path("admin-frontend/.env.local.sample")
    admin_env_local = Path("admin-frontend/.env.local")
    
    if not admin_env_sample.exists():
        print_error("admin-frontend/.env.local.sample file not found!")
    
    if not admin_env_local.exists():
        shutil.copy2(admin_env_sample, admin_env_local)
        print_success("Created admin-frontend/.env.local file")

def configure_env_variables():
    """Configure essential environment variables"""
    print_step("Configuring environment variables")
    
    env_file = Path(".env")
    if not env_file.exists():
        print_error(".env file not found!")
    
    # Read current env file
    with open(env_file, 'r') as f:
        env_lines = f.readlines()
    
    # Update essential variables
    env_vars = {}
    env_vars['ADMIN_PASSWORD'] = get_user_input("Enter admin password", "admin")
    env_vars['ANTHROPIC_API_KEY'] = get_user_input("Enter Anthropic API key (required for meeting notes)")
    
    # Update .env file
    with open(env_file, 'w') as f:
        for line in env_lines:
            if line.strip() and not line.startswith('#'):
                key = line.split('=')[0].strip()
                if key in env_vars:
                    f.write(f"{key}={env_vars[key]}\n")
                else:
                    f.write(line)
            else:
                f.write(line)

def start_docker():
    """Start Docker containers"""
    print_step("Starting Docker containers")
    
    try:
        # Build and start containers
        subprocess.run(["docker", "compose", "up", "--build", "-d"], check=True)
        print_success("Docker containers started successfully!")
        
        # Show container status
        print("\nContainer Status:")
        subprocess.run(["docker", "compose", "ps"], check=True)
        
        print(f"\n{Colors.GREEN}🎉 Setup complete! Your panottiServer is now running.{Colors.END}")
        print("\nAccess points:")
        print("- Admin Interface: http://localhost:3000")
        print("- API Endpoint: http://localhost:8000")
        print("\nTo view logs: docker compose logs -f")
        print("To stop: docker compose down")
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start Docker containers: {e}")

def main():
    """Main setup function"""
    # Ensure we're in the project root directory
    if not Path("docker-compose.yml").exists():
        print_error("Please run this script from the project root directory")
    
    check_docker()
    setup_plugin_configs()
    setup_env_files()
    configure_env_variables()
    start_docker()

if __name__ == "__main__":
    main()
