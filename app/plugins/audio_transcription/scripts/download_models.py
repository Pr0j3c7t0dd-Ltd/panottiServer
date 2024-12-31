#!/usr/bin/env python3
"""
Script to download Whisper models for offline use.
"""

import argparse
import os
import sys
from pathlib import Path

from faster_whisper import download_model


def get_project_root():
    """Get the project root directory."""
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent.parent.parent.parent


def download_whisper_model(model_name: str, output_dir: str):
    """
    Download a Whisper model for offline use.

    Args:
        model_name: Name of the model to download
            (e.g., 'base.en', 'small.en', 'medium.en', 'large-v2')
        output_dir: Directory to save the model
    """
    print(f"Downloading model '{model_name}' to {output_dir}...")
    try:
        # First ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download the model
        download_model(model_name, output_dir=output_dir)

        # Verify the download
        config_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_path):
            raise RuntimeError(f"Model files not found after download in {output_dir}")

        print(f"Successfully downloaded model '{model_name}' to {output_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Whisper models for offline use"
    )
    parser.add_argument(
        "--model",
        default="base.en",
        choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v2"],
        help="Model to download (default: base.en)",
    )
    parser.add_argument(
        "--output-dir", help="Output directory (default: PROJECT_ROOT/models/whisper)"
    )

    args = parser.parse_args()

    # Set default output directory if not specified
    if not args.output_dir:
        project_root = get_project_root()
        args.output_dir = os.path.join(project_root, "models", "whisper")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    download_whisper_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
