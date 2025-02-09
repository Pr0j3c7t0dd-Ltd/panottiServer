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

def check_ollama_setup():
    """Check if user wants to use local meeting note processing and setup Ollama"""
    if get_user_confirmation("\nDo you plan to process meeting notes locally on your machine?"):
        print("\n" + Colors.YELLOW + "⚠️  Important Note:" + Colors.END)
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

def setup_plugin_configs():
    """Copy plugin.yaml.example files to plugin.yaml for each plugin"""
    print_step("Setting up plugin configurations")
    
    root_dir = Path.cwd()
    plugin_dir = root_dir / "app" / "plugins"
    for example_file in plugin_dir.rglob("plugin.yaml.example"):
        target_file = example_file.parent / "plugin.yaml"
        if not target_file.exists():
            shutil.copy2(example_file, target_file)
            try:
                rel_path = target_file.relative_to(root_dir)
                print_success(f"Created {rel_path}")
            except ValueError:
                # Fallback to just the filename if relative_to fails
                print_success(f"Created {target_file.name}")

def setup_env_files():
    """Set up .env and admin frontend .env.local files"""
    print_step("Setting up environment files")
    
    # Main .env file
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print_error(".env.example file not found! Please ensure you're running this script from the project root.")
        sys.exit(1)
    
    if not env_file.exists():
        shutil.copy2(env_example, env_file)
        print_success("Created .env file from .env.example")
    
    # Admin frontend .env.local
    admin_env_sample = Path("admin-frontend/.env.local.sample")
    admin_env_local = Path("admin-frontend/.env.local")
    
    if not admin_env_sample.exists():
        print_warning("admin-frontend/.env.local.sample file not found - skipping admin frontend env setup")
        return
    
    if not admin_env_local.exists():
        shutil.copy2(admin_env_sample, admin_env_local)
        print_success("Created admin-frontend/.env.local file")

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

def configure_env_variables():
    """Configure essential environment variables"""
    print_step("Configuring environment variables")
    
    env_file = Path(".env")
    if not env_file.exists():
        print_error("No .env file found. Please run setup_env_files first.")
        sys.exit(1)
    
    try:
        # Read current env file
        with open(env_file, 'r') as f:
            env_lines = f.readlines()
        
        # Update essential variables
        env_vars = {}
        print("\nThe API_KEY should match the one set in your Panotti desktop app.")
        env_vars['API_KEY'] = get_user_input("Enter your API_KEY", "your_api_key_here")
        
        print("\nThe RECORDINGS_DIR should point to the same recordings directory set in your Panotti desktop app.")
        while True:
            host_recordings_dir = get_user_input("Enter the path to your recordings directory")
            recordings_path = Path(host_recordings_dir)
            
            # Convert to absolute path for Docker mounting
            try:
                abs_path = recordings_path.resolve(strict=True)
                if not abs_path.is_dir():
                    print_warning(f"Path exists but is not a directory: {abs_path}")
                    continue
                    
                # Check if directory is readable
                try:
                    next(abs_path.iterdir())
                except (PermissionError, StopIteration):
                    print_warning(f"Directory exists but may not be accessible: {abs_path}")
                    if not get_user_confirmation("Continue anyway?"):
                        continue
                
                host_recordings_dir = str(abs_path)
                break
            except FileNotFoundError:
                print_warning(f"Directory does not exist: {host_recordings_dir}")
                if get_user_confirmation("Create this directory?"):
                    try:
                        Path(host_recordings_dir).mkdir(parents=True)
                        host_recordings_dir = str(Path(host_recordings_dir).resolve())
                        break
                    except Exception as e:
                        print_warning(f"Failed to create directory: {e}")
                        continue
        
        print(f"\nUsing recordings directory: {host_recordings_dir}")
        print("Please ensure this path is shared with Docker:")
        print("Docker Desktop -> Settings -> Resources -> File Sharing")
        if not get_user_confirmation("Have you verified Docker has access to this directory?"):
            print_warning("Please share the directory with Docker and run this script again")
            sys.exit(1)
        
        # Store both the host path (for Docker mount) and container path (for FastAPI)
        env_vars['HOST_RECORDINGS_DIR'] = f'"{host_recordings_dir}"'
        env_vars['RECORDINGS_DIR'] = '"/recordings"'  # This is the path inside the container
        
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
        print_success("Environment variables updated successfully")
    except Exception as e:
        print_error(f"Failed to configure environment variables: {e}")
        sys.exit(1)

def create_ssl_directory():
    """Create SSL directory and generate self-signed certificates"""
    ssl_dir = Path("ssl")
    if not ssl_dir.exists():
        ssl_dir.mkdir()
        
    # Always regenerate certificates to ensure they exist
    print("Generating self-signed SSL certificates...")
    os.chdir(ssl_dir)
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
        print(Colors.BLUE + "   - 👉 Visit http://localhost:54790/  👈" + Colors.END)
        print("   - Default password: Pa55w0rd")
        print(f"   - API Endpoint: http://localhost:54789")
        print("\nTo view logs: docker compose logs -f")
        print("To stop: docker compose down")
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start Docker containers: {e}")

def main():
    """Main setup function"""
    if not os.path.isfile("docker-compose.yml"):
        print_error("Please run this script from the project root directory")
    
    check_docker()
    check_ollama_setup()
    setup_plugin_configs()
    setup_env_files()
    setup_admin_frontend()
    configure_env_variables()
    create_ssl_directory()
    start_docker()

if __name__ == "__main__":
    main()
