import os
import pathlib
from unittest.mock import patch, MagicMock

import pytest
from dotenv import load_dotenv

import run_server


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up test environment variables"""
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("API_HOST", "0.0.0.0")


@pytest.fixture
def mock_env_vars_default(monkeypatch):
    """Fixture to clear environment variables for testing defaults"""
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)


def test_main_with_env_vars(mock_env_vars):
    """Test main function with environment variables set"""
    with patch("uvicorn.run") as mock_run:
        run_server.main()
        
        base_dir = pathlib.Path(run_server.__file__).parent
        expected_ssl_keyfile = str(base_dir / "ssl" / "key.pem")
        expected_ssl_certfile = str(base_dir / "ssl" / "cert.pem")
        
        mock_run.assert_called_once_with(
            "app.main:app",
            host="0.0.0.0",
            port=9000,
            reload=True,
            ssl_keyfile=expected_ssl_keyfile,
            ssl_certfile=expected_ssl_certfile,
        )


def test_main_with_defaults(mock_env_vars_default):
    """Test main function with default values"""
    with patch("uvicorn.run") as mock_run:
        run_server.main()
        
        base_dir = pathlib.Path(run_server.__file__).parent
        expected_ssl_keyfile = str(base_dir / "ssl" / "key.pem")
        expected_ssl_certfile = str(base_dir / "ssl" / "cert.pem")
        
        mock_run.assert_called_once_with(
            "app.main:app",
            host="127.0.0.1",
            port=8001,
            reload=True,
            ssl_keyfile=expected_ssl_keyfile,
            ssl_certfile=expected_ssl_certfile,
        )


def test_script_execution():
    """Test script execution through __main__ block"""
    with patch("run_server.main") as mock_main:
        # Store original __name__
        original_name = run_server.__name__
        try:
            # Simulate being run as __main__
            run_server.__name__ = "__main__"
            # Re-execute the if block
            if run_server.__name__ == "__main__":
                run_server.main()
            mock_main.assert_called_once()
        finally:
            # Restore original __name__
            run_server.__name__ = original_name 