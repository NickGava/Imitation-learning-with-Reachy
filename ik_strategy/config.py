'''
config.py
=============================================================================
Usage:
    from config import DATA_ROOT
'''

from pathlib import Path

# Absolute path to the shared data folder.
# Resolved relative to this file's location, so it works regardless of the
# working directory from which a script is launched.
DATA_ROOT = Path(__file__).resolve().parent / "_data"