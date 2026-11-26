#!/usr/bin/env python3
"""Source-tree compatibility entry for the packaged smoke agent."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboodyssey.agents.random_agent import main

if __name__ == '__main__':
    main()
