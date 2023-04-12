import sys

import numpy as np
from tqdm import tqdm

import exploration
import file_writer as fw
import log_service
from predictors.initializer import PredictorsHandler
from predictors.models import Predictor
from search_space import SearchSpace
from utils import cell_pruning
from utils.cell_counter import CellCounter
from utils.cell_pruning import CellEncoding
from utils.config_dataclasses import SearchStrategyConfig, OthersConfig
from utils.func_utils import to_list_of_tuples
from utils.model_estimate import ModelEstimate
from utils.nn_utils import perform_global_memory_clear
from utils.rstr import rstr


class ControllerManager:
    '''
    Utility class to manage the accuracy and time predictors.

    Tasked with maintaining the state of the training schedule, keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume training.
    '''

    def __init__(self, search_space: SearchSpace, sstr_config: SearchStrategyConfig, others_config: OthersConfig,
                 predictor_handler: PredictorsHandler, current_b: int = 1):
        '''
        Manages the Controller network training and prediction process.
        '''
        self._logger = log_service.get_logger(__name__)
        self.search_space = search_space
        self.predictor_handler = predictor_handler

        self.B = search_space.B
        self.K = sstr_config.max_children
        self.ex = sstr_config.max_exploration_children
        self.current_b = current_b

        self.T = np.inf     # TODO: add it to sstr_config if actually necessary, for now we have never used the time constraint

        self.pareto_objectives = sstr_config.additional_pareto_objectives
        self.predictions_batch_size = others_config.predictions_batch_size
        self.acc_ensemble_units = others_config.accuracy_predictor_ensemble_units

        self.pnas_mode = others_config.pnas_mode

    def train_step(self):
        '''
        Train both accuracy and time predictors
        '''
        acc_predictor = self.predictor_handler.get_score_predictor(self.current_b)

        # train accuracy predictor with all data available
        if self.acc_ensemble_units > 1:
            acc_predictor.train_ensemble(log_service.build_path('csv', 'training_results.csv'), splits=self.acc_ensemble_units)
        # train an ensemble
        else:
            acc_predictor.train(log_service.build_path('csv', 'training_results.csv'))

        # train time predictor with new data
        if not self.pnas_mode and 'time' in self.pareto_objectives:
            csv_path = log_service.build_path('csv', 'training_time.csv')
            time_predictor = self.predictor_handler.get_time_predictor(self.current_b)
            time_predictor.train(csv_path)

        self._logger.info('Predictors training complete')

    def __generate_model_estimations(self, batched_models: 'list[tuple]', models_count: int, time_predictor: Predictor, acc_predictor: Predictor):
        model_estimations = []  # type: list[ModelEstimate]

        # use as total the actual predictions to make, but manually iterate on the batches with custom pbar update to reflect actual prediction speed
        pbar = tqdm(iterable=None,
                    unit='model', desc='Estimating models: ',
                    total=models_count,
                    file=sys.stdout)

        # iterate through all the possible cells for next B step and predict their score and time
        with pbar:
            for cells_batch in batched_models:
                estimated_scores = acc_predictor.predict_batch(cells_batch)

                # in PNAS mode only accuracy metric is needed
                if self.pnas_mode:
                    model_estimations.extend([ModelEstimate(cell_spec, score) for cell_spec, score in zip(cells_batch, estimated_scores)])
                # in POPNAS mode instead predict also training time and address additional Pareto problem metrics (like params)
                # Pareto objectives are defined in the configuration file, unused objectives are simply set to 0
                # to not alter the Pareto front construction
                else:
                    # TODO: conversion to features should be made in Predictor to make the interface consistent between NN and ML techniques
                    #  and make them fully swappable. A ML predictor class should be made in this case, since all models use the same feature set.
                    if 'time' in self.pareto_objectives:
                        batch_time_features = [self.predictor_handler.generate_cell_time_features(cell_spec) for cell_spec in cells_batch]
                        estimated_times = time_predictor.predict_batch(batch_time_features)
                    else:
                        estimated_times = [0] * len(cells_batch)

                    params_count = [self.predictor_handler.get_architecture_dag(cell_spec).get_total_params() for cell_spec in cells_batch] \
                        if 'params' in self.pareto_objectives else [0] * len(cells_batch)

                    # apply also time constraint
                    ests_in_time_limit = [ModelEstimate(cell_spec, score, time, params) for cell_spec, score, time, params
                                          in zip(cells_batch, estimated_scores, estimated_times, params_count) if time <= self.T]
                    model_estimations.extend(ests_in_time_limit)

                pbar.update(len(cells_batch))

        perform_global_memory_clear()

        return model_estimations

    def __build_pareto_front(self, model_estimations: 'list[ModelEstimate]'):
        '''
        Build the Pareto front from the predictions, limited to K elements.
        The Pareto front can be built only if using the time predictor (needs time estimation, not possible in PNAS mode).

        IMPORTANT: model_estimates must be sorted by score (max first).

        Args:
            model_estimations: list of cells, associated with the metrics targeted by Pareto optimization

        Returns:
            Pareto front, cell encodings inserted in Pareto front (can be used to prune equivalent models)
        '''
        self._logger.info('Building pareto front...')

        pareto_model_estimations = []
        pareto_eqv_cell_encodings = []

        for i, model_est in enumerate(model_estimations):
            model_est_cell_encoding = cell_pruning.CellEncoding(model_est.cell_spec)
            eqv_to_other_pareto_model = cell_pruning.is_model_equivalent_to_another(model_est_cell_encoding, pareto_eqv_cell_encodings)

            if not eqv_to_other_pareto_model and \
                    not any(model_est.is_dominated_by(other_est) for j, other_est in enumerate(model_estimations) if i != j):
                pareto_model_estimations.append(model_est)
                pareto_eqv_cell_encodings.append(model_est_cell_encoding)

            # algorithm has found K not equivalent Pareto optimal solutions, so it can return the solutions without wasting additional time
            if len(pareto_model_estimations) >= self.K:
                break

        self._logger.info('Pareto front built successfully')

        return pareto_model_estimations, pareto_eqv_cell_encodings

    def __build_exploration_pareto_front(self, model_estimations: 'list[ModelEstimate]', curr_model_reprs: 'list[CellEncoding]',
                                         op_exploration_set: set, input_exp_set: set):
        self._logger.info('Building exploration pareto front...')
        self._logger.info('Operators to explore: %s', rstr(op_exploration_set))
        self._logger.info('Inputs to explore: %s', rstr(input_exp_set))

        exp_cell_counter = CellCounter(input_exp_set, op_exploration_set)
        exploration_pareto_front = []
        pruned_count = 0

        # search the first element not already inserted in the standard Pareto front which satisfies the score threshold
        # if there is none (shouldn't be possible), then the function terminates immediately, returning an empty exploration Pareto front
        try:
            model_estimation = next(model_est
                                    for model_est in model_estimations
                                    if exploration.has_sufficient_exploration_score(model_est, exp_cell_counter) and
                                    not cell_pruning.is_model_equivalent_to_another(cell_pruning.CellEncoding(model_est.cell_spec), curr_model_reprs))

            exploration_pareto_front.append(model_estimation)
            curr_model_reprs.append(cell_pruning.CellEncoding(model_estimation.cell_spec))
            exp_cell_counter.update_from_cell_spec(model_estimation.cell_spec)
        except StopIteration:
            self._logger.info('No element satisfied the exploration score threshold, the exploration Pareto front is empty')
            return [], curr_model_reprs

        for i, model_est in enumerate(model_estimations):
            # less expected training time than the last pareto element
            if not any(model_est.is_dominated_by(epf_elem) for epf_elem in exploration_pareto_front) and \
                    exploration.has_sufficient_exploration_score(model_est, exp_cell_counter):
                cell_repr = cell_pruning.CellEncoding(model_est.cell_spec)

                # existing_model_reprs contains the pareto front and exploration cells will be progressively added to it
                # add model to the exploration front only if not equivalent to any model in both the standard pareto front and exploration front
                if not cell_pruning.is_model_equivalent_to_another(cell_repr, curr_model_reprs):
                    exploration_pareto_front.append(model_est)
                    curr_model_reprs.append(cell_repr)

                    exp_cell_counter.update_from_cell_spec(model_est.cell_spec)
                else:
                    pruned_count += 1

            # stop searching if we have reached <ex> elements in the exploration Pareto front
            if len(exploration_pareto_front) >= self.ex:
                break

        self._logger.info('Pruned %d equivalent models while building exploration pareto front', pruned_count)
        return exploration_pareto_front, curr_model_reprs

    def update_step(self):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        if self.current_b >= self.B:
            self._logger.info('No more updates necessary as target B has been reached!')
            return

        time_predictor = self.predictor_handler.get_time_predictor(self.current_b)
        acc_predictor = self.predictor_handler.get_score_predictor(self.current_b)

        self.current_b += 1
        # closure that returns the model generator for the current generation step
        generate_models = self.search_space.perform_children_expansion(self.current_b)

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
        fw.save_predictions_to_csv(model_estimations, f'predictions_B{self.current_b}.csv')

        self._logger.info('Models performance estimation completed')

        # The Pareto front is built only in POPNAS, while it is skipped in PNAS mode
        # Same for exploration Pareto front, if needed
        if not self.pnas_mode:
            pareto_front, existing_model_reprs = self.__build_pareto_front(model_estimations)
            fw.save_predictions_to_csv(pareto_front, f'pareto_front_B{self.current_b}.csv')

            # TODO: already limited right now
            # limit the Pareto front to K elements if necessary
            children_limit = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))
            pareto_front = pareto_front[:children_limit]
            existing_model_reprs = existing_model_reprs[:children_limit]

            # exploration step, avoid it if expanding cells with last block
            # or if there is no need for exploration at all (no input or op underused)
            op_exp, input_exp = exploration.compute_exploration_value_sets(pareto_front, self.search_space, self.current_b)

            if self.ex > 0 and self.current_b < self.B and (len(op_exp) > 0 or len(input_exp) > 0):
                exploration_pareto_front, existing_model_reprs = self.__build_exploration_pareto_front(model_estimations, existing_model_reprs,
                                                                                                       op_exp, input_exp)

                fw.save_predictions_to_csv(exploration_pareto_front, f'exploration_pareto_front_B{self.current_b}.csv')

                self.search_space.exploration_front = [child.cell_spec for child in exploration_pareto_front]
                self._logger.info('Exploration pareto front built successfully')
            else:
                self.search_space.exploration_front = []
                self._logger.info('No exploration necessary for this step')

            # take the cell specifications of Pareto elements, since they must be saved in the search space class
            children = [child.cell_spec for child in pareto_front]
        else:
            # just a renaming to integrate with existent code below, it's not a pareto front in this case!
            pareto_front = model_estimations

            # limit the Pareto front to K elements if necessary
            children_limit = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))

            # remove equivalent models, it is not done yet if running in pnas mode (children must be saved in the search space class)
            models = [est.cell_spec for est in pareto_front]
            children, pruned_count = cell_pruning.prune_equivalent_cell_models(models, children_limit)
            self._logger.info('Pruned %d equivalent models while selecting CNN children', pruned_count)

        fw.append_cell_spec_to_csv(children + self.search_space.exploration_front)

        # save these children for next round
        # exploration networks are saved in another separate variable (state_space.exploration_front)
        self.search_space.children = children
