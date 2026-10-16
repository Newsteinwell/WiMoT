"""Summary measures of a stage-wise test-accuracy matrix (row k: accuracies on the domains seen after update k).

  AFA   average final accuracy over all domains (last row)
  AFR   average forgetting: mean over earlier domains of (best accuracy after learning it, before the last update) minus final
  new   plasticity: mean over updates k >= 2 of the accuracy on the domain just learned (diagonal, first domain excluded)
  old   retention: mean accuracy on the earlier domains after the last update (last row without its last entry)

AFA and AFR are those of ``wimot.common.metrics``; ``new`` and ``old`` are those of scripts/plasticity_stability.py.
"""
import numpy as np

from .common import metrics


def summary_measures(matrix):
    a = np.array([[np.nan if v is None else v for v in row] for row in matrix], dtype=float)
    t = len(a); base = metrics(matrix)
    return dict(AFA=base['AFA'], AFR=base['AFR'], new=float(np.mean([a[k, k] for k in range(1, t)])),
                old=float(np.mean(a[-1, :t-1])))
