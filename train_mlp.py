#!/usr/bin/env python3
"""Train the weakly supervised MLP (features -> 64-d embedding + 6-class head) for one model.

Plain Python entry point: runs anywhere (laptop GPU, cluster, or CPU) with
no SLURM or cluster-specific assumptions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from athero.mlp.train import main

if __name__ == "__main__":
    main()
