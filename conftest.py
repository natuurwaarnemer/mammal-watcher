"""Pytest configuration: ensure project root is on sys.path.

Hierdoor kunnen de testen in tests/ de modules mammal_watcher en classifier
importeren die in de project-root staan.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
