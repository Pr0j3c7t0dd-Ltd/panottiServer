#!/usr/bin/env python3
"""
Build script for creating the panottiServer installer DMG.
Handles building the PyQt6 installer and packaging it into a DMG file.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

# ANSI Color codes for output
class Colors:
    HEADER = '\x1b[38;5;26m'     # Blue
    BLUE = '\x1b[38;5;26m'       # Blue
    GREEN = '\033[92m'      # Green
    YELLOW = '\033[93m'     # Yellow
    RED = '\033[91m'        # Red
    BOLD = '\033[1m'        # Bold
    END = '\033[0m'         # Reset

def print_step(emoji, text):
    """Print a build step with consistent formatting"""
    print("\n" + emoji + " " + f"{Colors.BLUE}{text}{Colors.END}")

def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}{text}{Colors.END}")

def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}{text}{Colors.END}")

def run_command(cmd, cwd=None):
    """Run a shell command and handle errors"""
    try:
        subprocess.run(cmd, check=True, shell=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Error running command: {cmd}")
        print_error(f"Error details: {e}")
        return False

def setup_installer_directory():
    """Create and set up the installer directory structure"""
    print_step("📁", "Setting up installer directory structure")
    
    installer_dir = Path("installer")
    if installer_dir.exists():
        shutil.rmtree(installer_dir)
    installer_dir.mkdir()
    
    # Copy necessary files
    shutil.copy("scripts/setup.py", installer_dir / "setup.py")
    
    # Create and setup assets directory
    setup_assets(installer_dir)
    
    # Create main.py with GUI installer code
    create_main_py(installer_dir)
    
    return installer_dir

def setup_assets(installer_dir):
    """Create and set up the assets directory with icon files"""
    print_step("🎨", "Setting up assets")
    
    assets_dir = installer_dir / "assets"
    assets_dir.mkdir()
    
    # Look for icon file in common locations
    icon_paths = [
        Path("assets/icon.png"),
        Path("assets/icon.jpg"),
        Path("assets/icon.icns"),
        Path("icon.png"),
        Path("icon.jpg"),
        Path("icon.icns"),
    ]
    
    icon_file = None
    for path in icon_paths:
        if path.exists():
            icon_file = path
            break
    
    if icon_file:
        print_step("🖼️", f"Found icon file: {icon_file}")
        
        # If icon is already in ICNS format, just copy it
        if icon_file.suffix == '.icns':
            shutil.copy(icon_file, assets_dir / "icon.icns")
            return assets_dir
        
        # Convert image to ICNS using iconutil (built into macOS)
        print_step("🔄", "Converting icon to ICNS format")
        
        # Create iconset directory
        iconset_dir = assets_dir / "icon.iconset"
        iconset_dir.mkdir()
        
        # Convert source image to PNG if needed
        if icon_file.suffix != '.png':
            if not run_command(f"sips -s format png {icon_file} --out {assets_dir}/temp.png"):
                print_error("Failed to convert image to PNG")
                return assets_dir
            icon_file = assets_dir / "temp.png"
        
        # Create various sizes required for ICNS
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for size in sizes:
            # Regular size
            run_command(f"sips -z {size} {size} {icon_file} --out {iconset_dir}/icon_{size}x{size}.png")
            # @2x size (Retina)
            if size <= 512:
                run_command(f"sips -z {size*2} {size*2} {icon_file} --out {iconset_dir}/icon_{size}x{size}@2x.png")
        
        # Convert iconset to ICNS
        run_command(f"iconutil -c icns {iconset_dir} -o {assets_dir}/icon.icns")
        
        # Clean up temporary files
        if (assets_dir / "temp.png").exists():
            (assets_dir / "temp.png").unlink()
        shutil.rmtree(iconset_dir)
        
    else:
        # Use default icon (blue P) if no icon file found
        print_step("ℹ️", "No icon file found, using default icon")
        
        # Create icon.svg content (simple P icon)
        icon_svg = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="1024" height="1024" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
    <rect width="1024" height="1024" rx="128" fill="#4A90E2"/>
    <path d="M312 256h200c110.457 0 200 89.543 200 200s-89.543 200-200 200H412v112H312V256zm100 300h100c55.228 0 100-44.772 100-100s-44.772-100-100-100H412v200z" fill="white"/>
</svg>
"""
        # Write SVG file
        with open(assets_dir / "icon.svg", "w") as f:
            f.write(icon_svg)
        
        # Convert SVG to PNG using sips
        run_command(f"rsvg-convert {assets_dir}/icon.svg -o {assets_dir}/temp.png")
        
        # Create iconset and convert to ICNS
        iconset_dir = assets_dir / "icon.iconset"
        iconset_dir.mkdir()
        
        # Create various sizes
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for size in sizes:
            # Regular size
            run_command(f"sips -z {size} {size} {assets_dir}/temp.png --out {iconset_dir}/icon_{size}x{size}.png")
            # @2x size (Retina)
            if size <= 512:
                run_command(f"sips -z {size*2} {size*2} {assets_dir}/temp.png --out {iconset_dir}/icon_{size}x{size}@2x.png")
        
        # Convert iconset to ICNS
        run_command(f"iconutil -c icns {iconset_dir} -o {assets_dir}/icon.icns")
        
        # Clean up temporary files
        (assets_dir / "temp.png").unlink()
        (assets_dir / "icon.svg").unlink()
        shutil.rmtree(iconset_dir)
    
    return assets_dir

def create_main_py(installer_dir):
    """Create the main.py file containing the GUI installer code"""
    print_step("📝", "Creating main.py installer script")
    
    code = """#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import (QApplication, QWizard, QWizardPage, QVBoxLayout, 
                           QLabel, QCheckBox, QProgressBar, QLineEdit, QGroupBox,
                           QPushButton, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import subprocess
from pathlib import Path
import os
import shutil

class SetupWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, options, config, install_path):
        super().__init__()
        self.options = options
        self.config = config
        self.install_path = install_path
        
    def run(self):
        try:
            total_steps = len([opt for opt in self.options.values() if opt])
            current_step = 0
            
            # First, copy the repository to the installation path
            self.progress.emit("Copying files to installation directory...", 10)
            
            # Get the repository root directory (two levels up from the installer executable)
            repo_root = Path(os.path.dirname(os.path.abspath(sys.argv[0]))).parent.parent
            
            # Create the installation directory if it doesn't exist
            os.makedirs(self.install_path, exist_ok=True)
            
            # Copy repository contents to installation directory
            for item in repo_root.iterdir():
                if item.name not in ['.git', 'installer', '.DS_Store']:
                    if item.is_dir():
                        shutil.copytree(item, Path(self.install_path) / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, Path(self.install_path) / item.name)
            
            # Change to installation directory
            os.chdir(self.install_path)
            
            # Run setup.py with selected options
            self.progress.emit("Running setup script...", 30)
            setup_script = os.path.join("scripts", "setup.py")
            if os.path.exists(setup_script):
                env = os.environ.copy()
                env.update({
                    'INSTALL_MODE': 'local',  # Signal to setup.py to use local paths
                    'API_KEY': self.config['api_key'],
                    'RECORDINGS_DIR': self.config['recordings_dir']
                })
                subprocess.run([sys.executable, setup_script], env=env, check=True)
                self.progress.emit("Setup completed successfully", 100)
            else:
                raise FileNotFoundError(f"Setup script not found at {setup_script}")
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class InstallWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("panottiServer Setup")
        self.install_path = None
        self.setup_pages()
        
    def setup_pages(self):
        # Installation Location page
        location = QWizardPage()
        location.setTitle("Installation Location")
        layout = QVBoxLayout()
        
        location_label = QLabel("Select where to install panottiServer:")
        self.location_input = QLineEdit()
        self.location_input.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        
        def browse():
            path = QFileDialog.getExistingDirectory(self, "Select Installation Directory")
            if path:
                self.install_path = path
                self.location_input.setText(path)
        
        browse_button.clicked.connect(browse)
        
        layout.addWidget(location_label)
        layout.addWidget(self.location_input)
        layout.addWidget(browse_button)
        
        # Make the Next button conditional on having a location
        location.registerField("install_location*", self.location_input)
        
        location.setLayout(layout)
        self.addPage(location)
        
        # Prerequisites page
        prereq = QWizardPage()
        prereq.setTitle("Prerequisites Check")
        layout = QVBoxLayout()
        
        # Ollama warning box
        warning_box = QGroupBox("⚠️ Required: Ollama Installation")
        warning_layout = QVBoxLayout()
        warning_layout.addWidget(QLabel("Before proceeding with the setup, you MUST:"))
        warning_layout.addWidget(QLabel("1. Download and install Ollama from: https://ollama.com/download"))
        warning_layout.addWidget(QLabel("2. Do NOT use Homebrew for Ollama installation"))
        warning_layout.addWidget(QLabel("3. Ensure Ollama is properly installed and running"))
        warning_layout.addWidget(QLabel("\\nDefault model: llama3.1:8b (will be downloaded during setup)"))
        warning_layout.addWidget(QLabel("Memory Requirements:"))
        warning_layout.addWidget(QLabel("- Minimum: 24GB RAM"))
        warning_layout.addWidget(QLabel("- Recommended: 32GB RAM"))
        warning_box.setLayout(warning_layout)
        layout.addWidget(warning_box)
        
        # Confirmation checkbox
        self.ollama_check = QCheckBox("I confirm that I have installed Ollama from ollama.com/download")
        layout.addWidget(self.ollama_check)
        
        # Make the Next button conditional on the checkbox
        prereq.registerField("ollama_installed*", self.ollama_check)
        
        prereq.setLayout(layout)
        self.addPage(prereq)
        
        # Component selection page
        components = QWizardPage()
        components.setTitle("Select Components to Setup")
        layout = QVBoxLayout()
        
        # Required Components Group
        required_group = QGroupBox("Required Components")
        required_layout = QVBoxLayout()
        self.homebrew_check = QCheckBox("Setup Homebrew (Required for system dependencies)")
        self.system_deps_check = QCheckBox("Setup system dependencies")
        self.python_check = QCheckBox("Setup Python 3.12")
        self.rust_check = QCheckBox("Setup Rust")
        self.poetry_check = QCheckBox("Setup Poetry")
        required_layout.addWidget(self.homebrew_check)
        required_layout.addWidget(self.system_deps_check)
        required_layout.addWidget(self.python_check)
        required_layout.addWidget(self.rust_check)
        required_layout.addWidget(self.poetry_check)
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)
        
        # Optional Components Group
        optional_group = QGroupBox("Optional Components")
        optional_layout = QVBoxLayout()
        self.venv_check = QCheckBox("Create virtual environment")
        self.whisper_check = QCheckBox("Download Whisper model")
        self.ssl_check = QCheckBox("Setup SSL certificates")
        self.node_check = QCheckBox("Setup Node.js")
        self.admin_frontend_check = QCheckBox("Setup admin frontend")
        optional_layout.addWidget(self.venv_check)
        optional_layout.addWidget(self.whisper_check)
        optional_layout.addWidget(self.ssl_check)
        optional_layout.addWidget(self.node_check)
        optional_layout.addWidget(self.admin_frontend_check)
        optional_group.setLayout(optional_layout)
        layout.addWidget(optional_group)
        
        components.setLayout(layout)
        self.addPage(components)
        
        # Configuration page
        config = QWizardPage()
        config.setTitle("Configuration")
        layout = QVBoxLayout()
        
        # API Key
        api_key_label = QLabel("API Key (should match your Panotti desktop app):")
        self.api_key_input = QLineEdit()
        layout.addWidget(api_key_label)
        layout.addWidget(self.api_key_input)
        
        # Recordings Directory
        recordings_dir_label = QLabel("Recordings Directory (should match your Panotti desktop app):")
        self.recordings_dir_input = QLineEdit()
        layout.addWidget(recordings_dir_label)
        layout.addWidget(self.recordings_dir_input)
        
        config.setLayout(layout)
        self.addPage(config)
        
        # Setup progress page
        progress = QWizardPage()
        progress.setTitle("Setting Up")
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Preparing setup...")
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        progress.setLayout(layout)
        self.addPage(progress)
        
    def get_options(self):
        return {
            'homebrew': self.homebrew_check.isChecked(),
            'system_deps': self.system_deps_check.isChecked(),
            'python': self.python_check.isChecked(),
            'rust': self.rust_check.isChecked(),
            'poetry': self.poetry_check.isChecked(),
            'venv': self.venv_check.isChecked(),
            'whisper': self.whisper_check.isChecked(),
            'ssl': self.ssl_check.isChecked(),
            'node': self.node_check.isChecked(),
            'admin_frontend': self.admin_frontend_check.isChecked(),
        }
        
    def get_config(self):
        return {
            'api_key': self.api_key_input.text(),
            'recordings_dir': self.recordings_dir_input.text(),
        }
        
    def perform_installation(self):
        if not self.install_path:
            self.status_label.setText("Error: No installation path selected")
            return
            
        options = self.get_options()
        config = self.get_config()
        
        self.worker = SetupWorker(options, config, self.install_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.installation_finished)
        self.worker.error.connect(self.installation_error)
        self.worker.start()
        
    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.progress_bar.setValue(value)
        
    def installation_finished(self):
        self.status_label.setText("Setup completed successfully!")
        
    def installation_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")

def main():
    # Enable high DPI scaling
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    wizard = InstallWizard()
    wizard.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
"""
    
    with open(installer_dir / "main.py", "w") as f:
        f.write(code)
    
    # Make it executable
    os.chmod(installer_dir / "main.py", 0o755)

def create_installer_requirements(installer_dir):
    """Create requirements.txt for the installer"""
    print_step("📝", "Creating installer requirements")
    
    requirements = [
        "PyQt6==6.6.1",
        "pyinstaller==6.11.1",
        "requests>=2.31.0"  # For Ollama API checks
    ]
    
    with open(installer_dir / "requirements.txt", "w") as f:
        f.write("\n".join(requirements))

def setup_virtual_env(installer_dir):
    """Set up a virtual environment for building the installer"""
    print_step("🌐", "Setting up virtual environment")
    
    if not run_command(f"python3 -m venv {installer_dir}/.venv"):
        return False
        
    # Activate virtual environment and install requirements
    activate_cmd = f"source {installer_dir}/.venv/bin/activate"
    install_cmd = f"pip install -r {installer_dir}/requirements.txt"
    
    return run_command(f"{activate_cmd} && {install_cmd}")

def build_installer(installer_dir):
    """Build the installer using PyInstaller"""
    print_step("🔨", "Building installer")
    
    activate_cmd = f"source {installer_dir}/.venv/bin/activate"
    build_cmd = (
        f"pyinstaller "
        f"--name=panottiServer-Installer "
        f"--windowed "
        f"--onefile "
        f"--clean "
        f"--icon=assets/icon.icns "
        f"--add-data 'setup.py:.' "
        f"--add-data 'assets:assets' "
        f"--codesign-identity=- "  # Ad-hoc signing
        f"--osx-bundle-identifier=com.panotti.server.installer "
        f"main.py"
    )
    
    return run_command(f"{activate_cmd} && cd {installer_dir} && {build_cmd}")

def create_dmg():
    """Create a DMG file from the built installer"""
    print_step("📀", "Creating DMG file")
    
    # Check if create-dmg is installed
    if not run_command("which create-dmg", cwd="installer"):
        print_step("📦", "Installing create-dmg")
        if not run_command("brew install create-dmg"):
            return False
    
    # Remove existing DMG if it exists
    dmg_path = Path("panottiServer-Installer.dmg")
    if dmg_path.exists():
        dmg_path.unlink()
    
    # Set the icon for the app bundle
    app_path = Path("installer/dist/panottiServer-Installer.app")
    resources_path = app_path / "Contents/Resources"
    icon_path = Path("installer/assets/icon.icns")
    
    # Ensure the Resources directory exists
    if not resources_path.exists():
        resources_path.mkdir(parents=True, exist_ok=True)
    
    # Copy the icon file
    if icon_path.exists():
        try:
            shutil.copy2(icon_path, resources_path / "icon-windowed.icns")
            # Also copy it as the standard app icon
            shutil.copy2(icon_path, resources_path / "icon.icns")
        except Exception as e:
            print_error(f"Failed to copy icon file: {e}")
            return False
    else:
        print_error(f"Icon file not found at {icon_path}")
        return False
    
    # Create DMG in root directory with simpler layout
    dmg_cmd = (
        "create-dmg "
        "--volname 'panottiServer Installer' "
        "--window-pos 200 120 "
        "--window-size 600 400 "
        "--icon-size 128 "
        "--icon 'panottiServer-Installer.app' 300 200 "
        "--hide-extension 'panottiServer-Installer.app' "
        "--app-drop-link 0 0 "  # Add this back to show the Applications folder link
        "--no-internet-enable "
        "../panottiServer-Installer.dmg "  # Output to root directory
        "dist/panottiServer-Installer.app"
    )
    
    return run_command(dmg_cmd, cwd="installer")

def cleanup():
    """Clean up build artifacts"""
    print_step("🧹", "Cleaning up build artifacts")
    
    paths_to_clean = [
        "installer/build",
        "installer/__pycache__",
        "installer/*.spec"
    ]
    
    for path in paths_to_clean:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        except Exception as e:
            print_error(f"Error cleaning up {path}: {e}")

    # Remove temporary DMG files generated by create-dmg
    for tmp_dmg in Path('.').glob('rw.*.panottiServer-Installer.dmg'):
        try:
            print_step("🧹", f"Removing temporary DMG file {tmp_dmg}")
            tmp_dmg.unlink()
        except Exception as e:
            print_error(f"Error removing temporary DMG file {tmp_dmg}: {e}")

def main():
    """Main build process"""
    try:
        print_step("🚀", "Starting panottiServer Installer build process")
        
        # Setup installer directory
        installer_dir = setup_installer_directory()
        if not installer_dir:
            sys.exit(1)
        
        # Create requirements file
        create_installer_requirements(installer_dir)
        
        # Setup virtual environment
        if not setup_virtual_env(installer_dir):
            sys.exit(1)
        
        # Build the installer
        if not build_installer(installer_dir):
            sys.exit(1)
        
        # Create DMG
        if not create_dmg():
            sys.exit(1)
        
        # Clean up
        cleanup()
        
        print_success("\n✨ Build completed successfully!")
        print_success("DMG file created at: panottiServer-Installer.dmg")  # Updated success message
        
    except KeyboardInterrupt:
        print_error("\n❌ Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 