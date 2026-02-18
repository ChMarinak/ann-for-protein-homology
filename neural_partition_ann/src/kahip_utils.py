# kahip_utils.py

import kahip
import numpy as np

def partition_graph(vwgt, xadj, adjncy, adjcwgt,
                    nblocks=100, imbalance=0.03,
                    seed=1, mode=2, suppress_output=True):
    """
    Partition a weighted, undirected graph using KaHIP.
    """

    # Ensure inputs are numpy arrays of integer type
    vwgt = np.array(vwgt, dtype=np.int32)
    xadj = np.array(xadj, dtype=np.int32)
    adjncy = np.array(adjncy, dtype=np.int32)
    adjcwgt = np.array(adjcwgt, dtype=np.int32)

    edgecut, blocks = kahip.kaffpa(
        vwgt,
        xadj,
        adjcwgt,
        adjncy,
        nblocks,       # number of partitions
        imbalance,     # allowed imbalance
        True,          # suppress_output
        1,             # seed
        mode           # mode (0,1,2)
    )
    blocks = np.array(blocks, dtype=np.int32)
    return blocks, edgecut
