import csv
from typing import Callable

import numpy as np
from tqdm import tqdm

import exploration
import log_service
from encoder import SearchSpace
from predictors import Predictor
from utils import cell_pruning
from utils.cell_counter import CellCounter
from utils.cell_pruning import CellEncoding
from utils.feature_utils import generate_time_features
from utils.func_utils import to_list_of_tuples
from utils.graph_generator import GraphGenerator
from utils.model_estimate import ModelEstimate
from utils.rstr import rstr


def _save_children_specs_to_file(children_specs: list):
    '''
    Append all cell specifications, that will be trained in the next step, in the children.csv file.

    Args:
        children_specs: all cells to train in next step (Pareto front + exploration Pareto front)
    '''
    with open(log_service.build_path('csv', 'children.csv'), mode='a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(children_specs)


def _save_exploration_pareto_front_to_file(exploration_pareto_front: 'list[ModelEstimate]', current_b: int):
    ''' Save exploration Pareto front to csv file. '''
    with open(log_service.build_path('csv', f'exploration_pareto_front_B{current_b}.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ModelEstimate.get_csv_headers())
        writer.writerows(map(lambda est: est.to_csv_array(), exploration_pareto_front))


def _save_pareto_front_to_file(pareto_front: 'list[ModelEstimate]', current_b: int):
    with open(log_service.build_path('csv', f'pareto_front_B{current_b}.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ModelEstimate.get_csv_headers())
        writer.writerows(map(lambda est: est.to_csv_array(), pareto_front))


class ControllerManager:
    '''
    Utility class to manage the accuracy and time predictors.

    Tasked with maintaining the state of the training schedule, keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume training.
    '''

    def __init__(self, search_space: SearchSpace, get_acc_predictor: Callable[[int], Predictor], get_time_predictor: Callable[[int], Predictor],
                 acc_predictor_ensemble_units: int, graph_generator: GraphGenerator, B=5, K=256, ex=16, T=np.inf,
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
        self.graph_generator = graph_generator

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

    def __generate_model_estimations(self, batched_models: 'list[tuple]', models_count: int, time_predictor: Predictor, acc_predictor: Predictor):
        model_estimations = []  # type: list[ModelEstimate]

        # use as total the actual predictions to make, but manually iterate on the batches with custom pbar update to reflect actual prediction speed
        pbar = tqdm(iterable=None,
                    unit='model', desc='Estimating models: ',
                    total=models_count)

        # iterate through all the possible cells for next B step and predict their score and time
        with pbar:
            for cells_batch in batched_models:
                # TODO: conversion to features should be made in Predictor to make the interface consistent between NN and ML techniques
                #  and make them fully swappable. A ML predictor class should be made in this case, since all models use the same feature set.
                batch_time_features = None if self.pnas_mode else [generate_time_features(cell_spec, self.search_space) for cell_spec in cells_batch]

                estimated_times = [None] * len(cells_batch) if self.pnas_mode else time_predictor.predict_batch(batch_time_features)
                estimated_scores = acc_predictor.predict_batch(cells_batch)
                # params_count = [NetworkGraph(cell_spec).get_total_params() for cell_spec in cells_batch]

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

        return model_estimations

    def __build_pareto_front(self, model_estimations: 'list[ModelEstimate]'):
        '''
        Build the Pareto front from the predictions.
        The Pareto front can be built only if using the time predictor (needs time estimation, not possible in PNAS mode).

        Args:
            model_estimations: list of cells, associated with the metrics targeted by Pareto optimization

        Returns:
            Pareto front, cell encodings inserted in Pareto front (can be used to prune equivalent models)
        '''
        self._logger.info('Building pareto front...')
        # The process by putting the first model into pareto front (best score, ordered array), then comparing
        # the rest only by time because of ordering trick.
        pareto_front = [model_estimations[0]]

        # for eqv check purposes
        existing_model_reprs = [cell_pruning.CellEncoding(model_estimations[0].cell_spec)]
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
        return pareto_front, existing_model_reprs

    def __build_exploration_pareto_front(self, model_estimations: 'list[ModelEstimate]', existing_model_reprs: 'list[CellEncoding]',
                                         op_exploration_set: set, input_exp_set: set):
        self._logger.info('Building exploration pareto front...')
        self._logger.info('Operators to explore: %s', rstr(op_exploration_set))
        self._logger.info('Inputs to explore: %s', rstr(input_exp_set))

        exp_cell_counter = CellCounter(input_exp_set, op_exploration_set)
        exploration_pareto_front = []
        last_el_time = math.inf
        pruned_count = 0

        for model_est in model_estimations[1:]:
            # continue searching until we have <ex> elements in the exploration pareto front
            if len(exploration_pareto_front) == self.ex:
                break

            # less time than last pareto element
            if model_est.time < last_el_time \
                    and exploration.has_sufficient_exploration_score(model_est, exp_cell_counter, exploration_pareto_front):
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
        return exploration_pareto_front, existing_model_reprs

    def update_step(self):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        if self.current_b >= self.B:
            self._logger.info('No more updates necessary as target B has been reached!')
            return

        time_predictor = self.get_time_predictor(self.current_b)
        acc_predictor = self.get_acc_predictor(self.current_b)

        self.current_b += 1
        # closure that returns a function that returns the model generator for current generation step
        generate_models = self.search_space.prepare_intermediate_children(self.current_b)

        # TODO: leave eqv models in estimation and prune them when extrapolating pareto front, so that it prunes only the
        #  necessary ones and takes lot less time (instead of O(N^2) it becomes O(len(pareto)^2)). Now done in that way,
        #  if you can make it more performant prune them before the evaluations again.
        next_models = list(generate_models())

        # process the models to predict in batches, to vastly optimize the time needed to process them all
        batched_models = to_list_of_tuples(next_models, self.predictions_batch_size)

        # use the predictors to estimate all the relevant metrics used in the pareto front
        model_estimations = self.__generate_model_estimations(batched_models, len(next_models), time_predictor, acc_predictor)

        # sort the children according to their score
        model_estimations = sorted(model_estimations, key=lambda x: x.score, reverse=True)
        self.__write_predictions_on_csv(model_estimations)

        self._logger.info('Models evaluation completed')

        # Pareto front is built only in POPNAS, while it is skipped in PNAS mode
        # Same for exploration Pareto front, if it is needed
        if not self.pnas_mode:
            pareto_front, existing_model_reprs = self.__build_pareto_front(model_estimations)
            _save_pareto_front_to_file(pareto_front, self.current_b)

            # limit the Pareto front to K elements if necessary
            children_limit = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))
            pareto_front = pareto_front[:children_limit]
            existing_model_reprs = existing_model_reprs[:children_limit]

            # exploration step, avoid it if expanding cells with last block
            # or if there is no need for exploration at all (no input or op underused)
            op_exp, input_exp = exploration.compute_exploration_value_sets(pareto_front, self.search_space, self.current_b, self.B)

            if self.ex > 0 and self.current_b < self.B and (len(op_exp) > 0 or len(input_exp) > 0):
                exploration_pareto_front, existing_model_reprs = self.__build_exploration_pareto_front(model_estimations, existing_model_reprs,
                                                                                                       op_exp, input_exp)

                _save_exploration_pareto_front_to_file(exploration_pareto_front, self.current_b)

                self.search_space.exploration_front = [child.cell_spec for child in exploration_pareto_front]
                self._logger.info('Exploration pareto front built successfully')
            else:
                self.search_space.exploration_front = []
                self._logger.info('No exploration necessary for this step')

            # take the cell specifications of Pareto elements, since they must be saved in the search space class
            children = [child.cell_spec for child in pareto_front]
        else:
            # just a rename to integrate with existent code below, it's not a pareto front in this case!
            pareto_front = model_estimations

            # limit the Pareto front to K elements if necessary
            children_limit = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))

            # remove equivalent models, not done already if running in pnas mode (children must be saved in the search space class)
            models = [est.cell_spec for est in pareto_front]
            children, pruned_count = cell_pruning.prune_equivalent_cell_models(models, children_limit)
            self._logger.info('Pruned %d equivalent models while selecting CNN children', pruned_count)

        _save_children_specs_to_file(children + self.search_space.exploration_front)

        # save these children for next round
        # exploration networks are saved in another separate variable (state_space.exploration_front)
        self.search_space.children = children
