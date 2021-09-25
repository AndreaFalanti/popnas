import numpy as np
import pandas as pd

from utils.func_utils import compute_spearman_rank_correlation_coefficient


# See also: https://catboost.ai/docs/concepts/python-usages-examples.html#custom-loss-function-eval-metric
class CatBoostEvalMetricSpearman:
    def get_final_error(self, error, weight):
        ''' Returns final value of metric based on error and weight '''
        return error

    def is_max_optimal(self):
        ''' Returns whether great values of metric are better '''
        return True

    def evaluate(self, approxes, target, weight):
        '''
        approxes is a list of indexed containers (containers with only __len__ and __getitem__ defined), one container per approx dimension.
        Each container contains floats.
        weight is a one dimensional indexed container.
        target is a one dimensional indexed container.

        weight parameter can be None.
        Returns pair (error, weights sum)
        '''
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        df_data = {
            'preds': np.array(approxes[0]),
            'target': np.array(target)
        }
        df = pd.DataFrame.from_dict(df_data)

        # weight sum not necessary, so set to 1
        return compute_spearman_rank_correlation_coefficient(df, 'target', 'preds'), 1
