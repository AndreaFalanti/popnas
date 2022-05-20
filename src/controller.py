import csv
import math
from collections import Counter
from typing import Callable

import numpy as np
from tqdm import tqdm

import log_service
from encoder import SearchSpace
from predictors import Predictor
from utils import cell_pruning
from utils.cell_counter import CellCounter
from utils.feature_utils import generate_time_features
from utils.func_utils import get_valid_inputs_for_block_size, to_list_of_tuples
from utils.rstr import rstr


class ModelEstimate:
    '''
    Helper class, basically a struct with a function to convert into array for csv saving
    '''

    def __init__(self, cell_spec, score, time):
        self.cell_spec = cell_spec
        self.score = score
        self.time = time

    def to_csv_array(self):
        cell_structure = f"[{';'.join(map(lambda el: str(el), self.cell_spec))}]"
        return [self.time, self.score, cell_structure]

    @staticmethod
    def get_csv_headers():
        return ['time', 'val accuracy', 'cell structure']


class ControllerManager:
    '''
    Utility class to manage the accuracy and time predictors.

    Tasked with maintaining the state of the training schedule, keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume training.
    '''

    def __init__(self, search_space: SearchSpace, get_acc_predictor: Callable[[int], Predictor], get_time_predictor: Callable[[int], Predictor],
                 acc_predictor_ensemble_units: int, B=5, K=256, ex=16, T=np.inf,
                 current_b: int = 1, predictions_batch_size: int = 16, pnas_mode: bool = False):
        '''
        Manages the Controller network training and prediction process.

        Args:
            search_space: completely defined search space.
            B: depth of progression.
            K: maximum number of children model trained per level of depth.
            T: maximum training time.
            pnas_mode: if True, do not build a regressor to estimate time. Use only LSTM controller,
                like original PNAS.
        '''
        self._logger = log_service.get_logger(__name__)
        self.search_space = search_space

        self.B = B
        self.K = K
        self.ex = ex
        self.T = T
        self.current_b = current_b
        self.predictions_batch_size = predictions_batch_size
        self.acc_ensemble_units = acc_predictor_ensemble_units

        self.get_time_predictor = get_time_predictor
        self.get_acc_predictor = get_acc_predictor

        self.pnas_mode = pnas_mode

    def train_step(self):
        '''
        Train both accuracy and time predictors
        '''
        acc_predictor = self.get_acc_predictor(self.current_b)

        # train accuracy predictor with all data available
        if self.acc_ensemble_units > 1:
            acc_predictor.train_ensemble(log_service.build_path('csv', 'training_results.csv'), splits=self.acc_ensemble_units)
        # train an ensemble
        else:
            acc_predictor.train(log_service.build_path('csv', 'training_results.csv'))

        # train time predictor with new data
        if not self.pnas_mode:
            csv_path = log_service.build_path('csv', 'training_time.csv')
            time_predictor = self.get_time_predictor(self.current_b)
            time_predictor.train(csv_path)

    def __write_predictions_on_csv(self, model_estimates):
        '''
        Write predictions on csv for further data analysis.

        Args:
            model_estimates (list[ModelEstimate]): [description]
        '''
        with open(log_service.build_path('csv', f'predictions_B{self.current_b}.csv'), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'val accuracy', 'cell structure'])
            writer.writerows(map(lambda model_est: model_est.to_csv_array(), model_estimates))

    def update_step(self):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        if self.current_b >= self.B:
            self._logger.info('No more updates necessary as max B has been reached!')
            return

        model_estimations = []  # type: list[ModelEstimate]
        time_predictor = self.get_time_predictor(self.current_b)
        acc_predictor = self.get_acc_predictor(self.current_b)

        self.current_b += 1
        # closure that returns a function that returns the model generator for current generation step
        generate_models = self.search_space.prepare_intermediate_children(self.current_b)

        next_models = list(generate_models())

        # process the models to predict in batches, to vastly optimize the time needed to process them all
        batched_models = to_list_of_tuples(next_models, self.predictions_batch_size)
        # use as total the actual predictions to make, but manually iterate on the batches with custom pbar update to reflect actual prediction speed
        pbar = tqdm(iterable=None,
                    unit='model', desc='Estimating models: ',
                    total=len(next_models))

        # iterate through all the possible cells for next B step and predict their score and time
        with pbar:
            for cells_batch in batched_models:
                batch_time_features = None if self.pnas_mode else [generate_time_features(cell_spec, self.search_space) for cell_spec in cells_batch]

                estimated_times = [None] * len(cells_batch) if self.pnas_mode else time_predictor.predict_batch(batch_time_features)
                estimated_scores = acc_predictor.predict_batch(cells_batch)

                # always preserve the child and its score in pnas mode
                if self.pnas_mode:
                    model_estimations.extend([ModelEstimate(cell_spec, score, time) for cell_spec, score, time
                                              in zip(cells_batch, estimated_scores, estimated_times)])
                # in popnas mode instead check that time estimation is < T (time threshold)
                else:
                    ests_in_time_limit = [ModelEstimate(cell_spec, score, time) for cell_spec, score, time
                                          in zip(cells_batch, estimated_scores, estimated_times) if time <= self.T]
                    model_estimations.extend(ests_in_time_limit)

                pbar.update(len(cells_batch))

        # sort the children according to their score
        model_estimations = sorted(model_estimations, key=lambda x: x.score, reverse=True)
        self.__write_predictions_on_csv(model_estimations)

        self._logger.info('Models evaluation completed')

        # start process by putting first model into pareto front (best score, ordered array),
        # then comparing the rest only by time because of ordering trick.
        # Pareto front can be built only if using regressor (needs time estimation, not possible in pnas mode)
        if not self.pnas_mode:
            self._logger.info('Building pareto front...')
            pareto_front = [model_estimations[0]]

            # for eqv check purposes
            existing_model_reprs = []
            pruned_count = 0

            for model_est in model_estimations[1:]:
                # less time than last pareto element
                if model_est.time < pareto_front[-1].time:
                    # check that model is not equivalent to another one present already in the pareto front
                    cell_repr = cell_pruning.CellEncoding(model_est.cell_spec)
                    if not cell_pruning.check_model_equivalence(cell_repr, existing_model_reprs):
                        pareto_front.append(model_est)
                        existing_model_reprs.append(cell_repr)
                    else:
                        pruned_count += 1

            self._logger.info('Pruned %d equivalent models while building pareto front', pruned_count)

            with open(log_service.build_path('csv', f'pareto_front_B{self.current_b}.csv'), mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(ModelEstimate.get_csv_headers())
                writer.writerows(map(lambda est: est.to_csv_array(), pareto_front))

            self._logger.info('Pareto front built successfully')

            # exploration step, avoid it if expanding cells with last block or if there is no need for exploration at all
            pareto_limit = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))
            op_exp, input_exp = self.compute_exploration_value_sets(pareto_front[:pareto_limit], self.search_space)

            if self.ex > 0 and self.current_b < self.B and (len(op_exp) > 0 or len(input_exp) > 0):
                self._logger.info('Building exploration pareto front...')
                self._logger.info('Operators to explore: %s', rstr(op_exp))
                self._logger.info('Inputs to explore: %s', rstr(input_exp))

                exp_cell_counter = CellCounter(input_exp, op_exp)
                exploration_pareto_front = []
                last_el_time = math.inf
                pruned_count = 0

                for model_est in model_estimations[1:]:
                    # continue searching until we have <ex> elements in the exploration pareto front
                    if len(exploration_pareto_front) == self.ex:
                        break

                    # less time than last pareto element
                    if model_est.time < last_el_time and self.has_sufficient_exploration_score(model_est, exp_cell_counter, exploration_pareto_front):
                        cell_repr = cell_pruning.CellEncoding(model_est.cell_spec)

                        # existing_model_reprs contains the pareto front and exploration cells will be progressively added to it
                        # add model to exploration front only if not equivalent to any model in both standard pareto front and exploration front
                        if not cell_pruning.check_model_equivalence(cell_repr, existing_model_reprs):
                            exploration_pareto_front.append(model_est)
                            existing_model_reprs.append(cell_repr)
                            last_el_time = model_est.time

                            exp_cell_counter.update_from_cell_spec(model_est.cell_spec)
                        else:
                            pruned_count += 1

                self._logger.info('Pruned %d equivalent models while building exploration pareto front', pruned_count)

                with open(log_service.build_path('csv', f'exploration_pareto_front_B{self.current_b}.csv'), mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(ModelEstimate.get_csv_headers())
                    writer.writerows(map(lambda est: est.to_csv_array(), exploration_pareto_front))

                self.search_space.exploration_front = [child.cell_spec for child in exploration_pareto_front]
                self._logger.info('Exploration pareto front built successfully')
            else:
                self.search_space.exploration_front = []
                self._logger.info('No exploration necessary for this step')
        else:
            # just a rename to integrate with existent code below, it's not a pareto front in this case!
            pareto_front = model_estimations

        # account for case where there are fewer children than K
        children_count = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))

        if not self.pnas_mode:
            # take only the K highest scoring children for next iteration + the elements of the exploration pareto front
            children = [child.cell_spec for child in pareto_front[:children_count]]
        else:
            # remove equivalent models, not done already if running in pnas mode
            models = [est.cell_spec for est in pareto_front]
            children, pruned_count = cell_pruning.prune_equivalent_cell_models(models, children_count)
            self._logger.info('Pruned %d equivalent models while selecting CNN children', pruned_count)

        cells_to_train = children + self.search_space.exploration_front
        with open(log_service.build_path('csv', 'children.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(cells_to_train)

        # save these children for next round
        # exploration networks are saved in another separate variable (state_space.exploration_front)
        self.search_space.children = children

    #   ///////////////////////////////////////////////////
    #  ///   EXPLORATION MECHANISM RELATED FUNCTIONS   ///
    # ///////////////////////////////////////////////////

    def compute_exploration_value_sets(self, pareto_front_models: 'list[ModelEstimate]', state_space: SearchSpace):
        valid_inputs = get_valid_inputs_for_block_size(state_space.input_values, self.current_b, self.B)
        valid_ops = state_space.operator_values
        cell_counter = CellCounter(valid_inputs, valid_ops)

        for model in pareto_front_models:
            cell_counter.update_from_cell_spec(model.cell_spec)

        op_usage_threshold = cell_counter.ops_total() / (len(valid_ops) * 5)
        input_usage_threshold = cell_counter.inputs_total() / (len(valid_inputs) * 5)

        op_exp = [key for key, val in cell_counter.op_counter.items() if val < op_usage_threshold]
        input_exp = [key for key, val in cell_counter.input_counter.items() if val < input_usage_threshold]

        return op_exp, input_exp

    def get_block_element_exploration_score(self, el, exploration_counter: Counter, total_count: int, bonus: bool):
        score = 0

        # el in exploration set (dict is initialized with valid keys)
        if el in exploration_counter.keys():
            score += 1

            # el underused condition (less than average). If only one element is present, it will be always True.
            if total_count == 0 or exploration_counter[el] <= (total_count / len(exploration_counter.keys())):
                score += 2

            if bonus:
                score += 1

        return score

    def has_sufficient_exploration_score(self, model_est: ModelEstimate, exp_cell_counter: CellCounter,
                                         exploration_pareto_front: 'list[ModelEstimate]'):
        exp_score = 0
        exp_inputs_total_count = exp_cell_counter.inputs_total()
        exp_ops_total_count = exp_cell_counter.ops_total()

        # give a bonus to the least searched set between inputs and operators (to both if equal)
        # in case one exploration set is empty, after first step the bonus will not be granted anymore.
        op_bonus = exp_ops_total_count <= exp_inputs_total_count
        input_bonus = exp_inputs_total_count <= exp_ops_total_count

        for in1, op1, in2, op2 in model_est.cell_spec:
            exp_score += self.get_block_element_exploration_score(in1, exp_cell_counter.input_counter, exp_inputs_total_count, input_bonus)
            exp_score += self.get_block_element_exploration_score(in2, exp_cell_counter.input_counter, exp_inputs_total_count, input_bonus)
            exp_score += self.get_block_element_exploration_score(op1, exp_cell_counter.op_counter, exp_ops_total_count, op_bonus)
            exp_score += self.get_block_element_exploration_score(op1, exp_cell_counter.op_counter, exp_ops_total_count, op_bonus)

        # additional conditions for pareto variety (float values, 1 point every difference of 4% of accuracy or 10% of time difference
        # with previous pareto entry). Considered only for cells with elements in exploration sets, when exploration pareto front is not empty.
        if exp_score > 0 and len(exploration_pareto_front) > 0:
            exp_score += (1 - model_est.score / exploration_pareto_front[-1].score) / 0.04
            exp_score += (1 - model_est.time / exploration_pareto_front[-1].time) / 0.10

        # adapt threshold if one of the two sets is empty
        exp_score_threshold = 8 if (exp_cell_counter.ops_keys_len() > 0 and exp_cell_counter.inputs_keys_len() > 0) else 4
        return exp_score >= exp_score_threshold
