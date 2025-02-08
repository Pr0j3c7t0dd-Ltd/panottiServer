#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path
import shutil
from PyQt6 import QtWidgets, QtCore

class SetupWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    
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

class InstallWizard(QtWidgets.QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("panottiServer Setup")
        self.install_path = None
        self.setup_pages()
        
    def setup_pages(self):
        # Installation Location page
        location = QtWidgets.QWizardPage()
        location.setTitle("Installation Location")
        layout = QtWidgets.QVBoxLayout()
        
        location_label = QtWidgets.QLabel("Select where to install panottiServer:")
        self.location_input = QtWidgets.QLineEdit()
        self.location_input.setReadOnly(True)
        browse_button = QtWidgets.QPushButton("Browse...")
        
        def browse():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Installation Directory")
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
        prereq = QtWidgets.QWizardPage()
        prereq.setTitle("Prerequisites Check")
        layout = QtWidgets.QVBoxLayout()
        
        # Ollama warning box
        warning_box = QtWidgets.QGroupBox("⚠️ Required: Ollama Installation")
        warning_layout = QtWidgets.QVBoxLayout()
        warning_layout.addWidget(QtWidgets.QLabel("Before proceeding with the setup, you MUST:"))
        warning_layout.addWidget(QtWidgets.QLabel("1. Download and install Ollama from: https://ollama.com/download"))
        warning_layout.addWidget(QtWidgets.QLabel("2. Do NOT use Homebrew for Ollama installation"))
        warning_layout.addWidget(QtWidgets.QLabel("3. Ensure Ollama is properly installed and running"))
        warning_layout.addWidget(QtWidgets.QLabel("\nDefault model: llama3.1:8b (will be downloaded during setup)"))
        warning_layout.addWidget(QtWidgets.QLabel("Memory Requirements:"))
        warning_layout.addWidget(QtWidgets.QLabel("- Minimum: 24GB RAM"))
        warning_layout.addWidget(QtWidgets.QLabel("- Recommended: 32GB RAM"))
        warning_box.setLayout(warning_layout)
        layout.addWidget(warning_box)
        
        # Confirmation checkbox
        self.ollama_check = QtWidgets.QCheckBox("I confirm that I have installed Ollama from ollama.com/download")
        layout.addWidget(self.ollama_check)
        
        # Make the Next button conditional on the checkbox
        prereq.registerField("ollama_installed*", self.ollama_check)
        
        prereq.setLayout(layout)
        self.addPage(prereq)
        
        # Component selection page
        components = QtWidgets.QWizardPage()
        components.setTitle("Select Components to Setup")
        layout = QtWidgets.QVBoxLayout()
        
        # Required Components Group
        required_group = QtWidgets.QGroupBox("Required Components")
        required_layout = QtWidgets.QVBoxLayout()
        self.homebrew_check = QtWidgets.QCheckBox("Setup Homebrew (Required for system dependencies)")
        self.system_deps_check = QtWidgets.QCheckBox("Setup system dependencies")
        self.python_check = QtWidgets.QCheckBox("Setup Python 3.12")
        self.rust_check = QtWidgets.QCheckBox("Setup Rust")
        self.poetry_check = QtWidgets.QCheckBox("Setup Poetry")
        required_layout.addWidget(self.homebrew_check)
        required_layout.addWidget(self.system_deps_check)
        required_layout.addWidget(self.python_check)
        required_layout.addWidget(self.rust_check)
        required_layout.addWidget(self.poetry_check)
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)
        
        # Optional Components Group
        optional_group = QtWidgets.QGroupBox("Optional Components")
        optional_layout = QtWidgets.QVBoxLayout()
        self.venv_check = QtWidgets.QCheckBox("Create virtual environment")
        self.whisper_check = QtWidgets.QCheckBox("Download Whisper model")
        self.ssl_check = QtWidgets.QCheckBox("Setup SSL certificates")
        self.node_check = QtWidgets.QCheckBox("Setup Node.js")
        self.admin_frontend_check = QtWidgets.QCheckBox("Setup admin frontend")
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
        config = QtWidgets.QWizardPage()
        config.setTitle("Configuration")
        layout = QtWidgets.QVBoxLayout()
        
        # API Key
        api_key_label = QtWidgets.QLabel("API Key (should match your Panotti desktop app):")
        self.api_key_input = QtWidgets.QLineEdit()
        layout.addWidget(api_key_label)
        layout.addWidget(self.api_key_input)
        
        # Recordings Directory
        recordings_dir_label = QtWidgets.QLabel("Recordings Directory (should match your Panotti desktop app):")
        self.recordings_dir_input = QtWidgets.QLineEdit()
        layout.addWidget(recordings_dir_label)
        layout.addWidget(self.recordings_dir_input)
        
        config.setLayout(layout)
        self.addPage(config)
        
        # Setup progress page
        progress = QtWidgets.QWizardPage()
        progress.setTitle("Setting Up")
        layout = QtWidgets.QVBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.status_label = QtWidgets.QLabel("Preparing setup...")
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
    app = QtWidgets.QApplication(sys.argv)
    wizard = InstallWizard()
    wizard.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
