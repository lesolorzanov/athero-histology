#!/usr/bin/env python3
"""Segment new slides: 64-d embedding + kNN label transfer from the study tiles.

Plain Python entry point: runs anywhere (laptop GPU, cluster, or CPU) with
no SLURM or cluster-specific assumptions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from athero.annotate import main

if __name__ == "__main__":
    main()
