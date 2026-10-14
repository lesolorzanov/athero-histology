#!/usr/bin/env python3
"""Slide-level train/validation partition of the annotated slides, stratified by class prevalence.

Plain Python entry point: runs anywhere (laptop GPU, cluster, or CPU) with
no SLURM or cluster-specific assumptions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from partition.split import main

if __name__ == "__main__":
    main()
