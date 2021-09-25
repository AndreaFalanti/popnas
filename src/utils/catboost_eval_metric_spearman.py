import numpy as np


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

        # approxes is already a numpy array, but inside a tuple. target instead is fine (ndarray).
        preds = approxes[0]

        # spearman correlation coefficient computation, done entirely with numpy so that numba can compile this function
        cov_n0 = np.cov(target, preds, ddof=0)[0][1]

        s_x0 = np.std(target)
        s_y0 = np.std(preds)

        spearman_coeff = cov_n0 / (s_x0 * s_y0)

        # weight sum not necessary, so set to 1
        return spearman_coeff, 1
