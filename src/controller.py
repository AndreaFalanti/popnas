import csv
import math
from collections import Counter
from typing import Callable

import numpy as np
from tqdm import tqdm

import cell_pruning
import log_service
from encoder import StateSpace
from predictors import Predictor
from utils.cell_counter import CellCounter
from utils.feature_utils import generate_time_features
from utils.func_utils import get_valid_inputs_for_block_size
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

    def __init__(self, state_space: StateSpace, get_acc_predictor: Callable[[int], Predictor], get_time_predictor: Callable[[int], Predictor],
                 B=5, K=256, ex=16, T=np.inf, pnas_mode: bool = False):
        '''
        Manages the Controller network training and prediction process.

        Args:
            state_space: completely defined search space.
            B: depth of progression.
            K: maximum number of children model trained per level of depth.
            T: maximum training time.
            pnas_mode: if True, do not build a regressor to estimate time. Use only LSTM controller,
                like original PNAS.
        '''
        self._logger = log_service.get_logger(__name__)
        self.state_space = state_space

        self.B = B
        self.K = K
        self.ex = ex
        self.T = T
        self.actual_b = 1

        self.get_time_predictor = get_time_predictor
        self.get_acc_predictor = get_acc_predictor

        self.pnas_mode = pnas_mode

    def train_step(self, rewards):
        '''
        Train both accuracy and time predictors
        '''

        train_cells = self.state_space.children + self.state_space.exploration_front
        acc_predictor = self.get_acc_predictor(self.actual_b)

        # train accuracy predictor with new data
        dataset = list(zip(train_cells, rewards))
        acc_predictor.train(dataset)

        # train time predictor with new data
        if not self.pnas_mode:
            csv_path = log_service.build_path('csv', 'training_time.csv')
            time_predictor = self.get_time_predictor(self.actual_b)
            time_predictor.train(csv_path)

    def __write_predictions_on_csv(self, model_estimates):
        '''
        Write predictions on csv for further data analysis.

        Args:
            model_estimates (list[ModelEstimate]): [description]
        '''
        with open(log_service.build_path('csv', f'predictions_B{self.actual_b}.csv'), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'val accuracy', 'cell structure'])
            writer.writerows(map(lambda model_est: model_est.to_csv_array(), model_estimates))

    def update_step(self):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        if self.actual_b >= self.B:
            self._logger.info('No more updates necessary as max B has been reached!')
            return

        model_estimations = []  # type: list[ModelEstimate]
        time_predictor = self.get_time_predictor(self.actual_b)
        acc_predictor = self.get_acc_predictor(self.actual_b)

        self.actual_b += 1
        # closure that returns a function that returns the model generator for current generation step
        generate_models = self.state_space.prepare_intermediate_children(self.actual_b)

        # TODO: leave eqv models in estimation and prune them when extrapolating pareto front, so that it prunes only the
        #  necessary ones and takes lot less time (instead of O(N^2) it becomes O(len(pareto)^2)). Now done in that way,
        #  if you can make it more performant prune them before the evaluations again.
        next_models = list(generate_models())

        pbar = tqdm(iterable=next_models,
                    unit='model', desc='Estimating models: ',
                    total=len(next_models))

        # iterate through all the intermediate children (intermediate_child is a list of repeated (input,op,input,op) tuple blocks)
        for cell_spec in pbar:
            # TODO: conversion to features should be made in Predictor to make the interface consistent between NN and ML techniques
            #  and make them fully swappable. A ML predictor class should be made in this case, since all models use the same feature set.
            time_features = None if self.pnas_mode else generate_time_features(cell_spec, self.state_space)
            estimated_time = None if self.pnas_mode else time_predictor.predict(time_features)

            estimated_score = acc_predictor.predict(cell_spec)

            pbar.set_postfix({'time': estimated_time, 'score': estimated_score}, refresh=False)

            # always preserve the child and its score in pnas mode, otherwise check that time estimation is < T (time threshold)
            if self.pnas_mode or estimated_time <= self.T:
                model_estimations.append(ModelEstimate(cell_spec, estimated_score, estimated_time))

        # sort the children according to their score
        model_estimations = sorted(model_estimations, key=lambda x: x.score, reverse=True)
        self.__write_predictions_on_csv(model_estimations)

        self._logger.info('Model evaluation completed')

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

            with open(log_service.build_path('csv', f'pareto_front_B{self.actual_b}.csv'), mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(ModelEstimate.get_csv_headers())
                writer.writerows(map(lambda est: est.to_csv_array(), pareto_front))

            self._logger.info('Pareto front built successfully')

            # exploration step, avoid it if expanding cells with last block or if there is no need for exploration at all
            pareto_limit = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))
            op_exp, input_exp = self.compute_exploration_value_sets(pareto_front[:pareto_limit], self.state_space)

            if self.actual_b < self.B and (len(op_exp) > 0 or len(input_exp) > 0):
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

                with open(log_service.build_path('csv', f'exploration_pareto_front_B{self.actual_b}.csv'), mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(ModelEstimate.get_csv_headers())
                    writer.writerows(map(lambda est: est.to_csv_array(), exploration_pareto_front))

                self.state_space.exploration_front = [child.cell_spec for child in exploration_pareto_front]
                self._logger.info('Exploration pareto front built successfully')
            else:
                self.state_space.exploration_front = []
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

        cells_to_train = children + self.state_space.exploration_front
        with open(log_service.build_path('csv', 'children.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(cells_to_train)

        # save these children for next round
        # exploration networks are saved in another separate variable (state_space.exploration_front)
        self.state_space.update_children(children)

    #   ///////////////////////////////////////////////////
    #  ///   EXPLORATION MECHANISM RELATED FUNCTIONS   ///
    # ///////////////////////////////////////////////////

    def compute_exploration_value_sets(self, pareto_front_models: 'list[ModelEstimate]', state_space: StateSpace):
        valid_inputs = get_valid_inputs_for_block_size(state_space.input_values, self.actual_b, self.B)
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
