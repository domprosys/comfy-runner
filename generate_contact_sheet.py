#!/usr/bin/env python3
"""
Thin wrapper around the packaged contact sheet generator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from comfy_api.contact_sheet import main


if __name__ == "__main__":
    main()
