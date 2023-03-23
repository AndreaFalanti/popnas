'''
Currently contains only experimental losses aimed for improving ranking in the predictors.
From some ablation studies, it is difficult to verify if there is an actual benefit in using them, and they could be prone to bugs.
Use them with caution.
'''
from .mse_hinge_loss import PairwiseHingeFromPred, MSEWithPairwiseHinge
from .spearman_loss import Spearman, MSEWithSpearman
from .squared_rank_error_loss import SquaredRankError, MSEWithSRE

__all__ = ['PairwiseHingeFromPred', 'MSEWithSRE', 'SquaredRankError', 'MSEWithSRE', 'Spearman', 'MSEWithSpearman']
