"""
Script to run the pairs trading strategy.
"""

import os
import sys

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Run the strategy
from src.run_strategy import main

if __name__ == '__main__':
    main()
