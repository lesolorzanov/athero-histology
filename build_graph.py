#!/usr/bin/env python3
"""Project features to 64-d, build the kNN graph, run Leiden, propagate labels (community majority).

Plain Python entry point: runs anywhere (laptop GPU, cluster, or CPU) with
no SLURM or cluster-specific assumptions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from athero.graph.run import main

if __name__ == "__main__":
    main()
