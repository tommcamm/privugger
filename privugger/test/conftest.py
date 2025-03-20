"""
Pytest configuration file for Privugger tests.

This file contains pytest fixtures and configuration for the test suite.
It helps with proper module imports and test discovery.
"""
import os
import sys
import pytest

# Make sure privugger package is importable regardless of how tests are run
# This ensures tests work with `python -m pytest` and direct `pytest` calls
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
