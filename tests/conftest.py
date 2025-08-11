"""
Configuration file for pytest.
This file ensures that the parent directory is in the Python path,
allowing tests to import modules from the parent directory.
"""

import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
