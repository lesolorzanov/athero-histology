#!/usr/bin/env python3
"""Evaluation on held-out validation tiles: 3-level confusion matrices and per-class AUROC.

Plain Python entry point: runs anywhere (laptop GPU, cluster, or CPU) with
no SLURM or cluster-specific assumptions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from athero.evaluation.metrics import main

if __name__ == "__main__":
    main()
