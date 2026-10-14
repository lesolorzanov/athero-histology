#!/usr/bin/env python3
"""Extract tile features from slide DZI tile folders with one foundation model (uni | uni2 | v2 | ciga | tc).

Plain Python entry point: runs anywhere (laptop GPU, cluster, or CPU) with
no SLURM or cluster-specific assumptions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from athero.features.extract import main

if __name__ == "__main__":
    main()
