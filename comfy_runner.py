#!/usr/bin/env python3
"""
Thin wrapper around the packaged runner.
"""

import sys
from pathlib import Path

# Ensure local package is importable without installation
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from comfy_api.runner import main


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    # Get workflow file from command line or use default
    if len(sys.argv) > 1:
        workflow_path = sys.argv[1]
    else:
        workflow_path = "video_wan2_2_5B_ti2v.json"

    run_workflow(workflow_path)
