import csv
from typing import Callable

import numpy as np
from tqdm import tqdm

import cell_pruning
import log_service
from encoder import StateSpace
from predictors import Predictor


class ControllerManager:
    '''
    Utility class to manage the accuracy and time predictors.

    Tasked with maintaining the state of the training schedule, keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume training.
    '''

    def __init__(self, state_space: StateSpace, checkpoint_B,
                 get_acc_predictor: Callable[[int], Predictor], get_time_predictor: Callable[[int], Predictor],
                 B=5, K=256, T=np.inf,
                 pnas_mode: bool = False, restore_controller: bool = False):
        '''
        Manages the Controller network training and prediction process.

        Args:
            state_space: completely defined search space.
            B: depth of progression.
            K: maximum number of children model trained per level of depth.
            T: maximum training time.
            pnas_mode: if True, do not build a regressor to estimate time. Use only LSTM controller,
                like original PNAS.
            restore_controller: flag whether to restore a pre-trained RNN controller
                upon construction.
        '''
        self._logger = log_service.get_logger(__name__)
        self.state_space = state_space

        self.B = B
        self.K = K
        self.T = T

        self.get_time_predictor = get_time_predictor
        self.get_acc_predictor = get_acc_predictor

        self.pnas_mode = pnas_mode
        self.restore_controller = restore_controller

        self.build_regressor_config = True

        # restore controller
        # TODO: surely not working by beginning, it used csv files that don't exist!
        if self.restore_controller:
            # TODO
            raise NotImplementedError('Restoring controller is not implemented right now')
        else:
            self.actual_b = 1

    def train_step(self, rewards):
        '''
        Train both accuracy and time predictors
        '''

        train_cells = self.state_space.children
        acc_predictor = self.get_acc_predictor(self.actual_b)

        # train accuracy predictor with new data
        dataset = list(zip(train_cells, rewards))
        acc_predictor.train(dataset)

        # train time predictor with new data
        csv_path = log_service.build_path('csv', 'training_time.csv')
        time_predictor = self.get_time_predictor(self.actual_b)
        time_predictor.train(csv_path)

    def get_time_features_from_cell_spec(self, child_spec: list):
        '''
        Produce the time features to be used in time predictor

        Args:
            child_spec (list[str]): model encoding

        Returns:
            (list): features list
        '''
        # regressor uses dynamic reindex for operations, instead of categorical
        encoded_child = self.state_space.encode_cell_spec(child_spec, op_enc_name='dynamic_reindex')

        # add missing blocks num feature (see training_time.csv, all columns except time are needed)
        regressor_features = [self.actual_b] + encoded_child

        # complete features with missing blocks (0 is for null)
        return regressor_features + [0, 0, 0, 0] * (self.B - self.actual_b)

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
        if self.actual_b + 1 <= self.B:
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
                time_features = None if self.pnas_mode else self.get_time_features_from_cell_spec(cell_spec)
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
                    if model_est.time <= pareto_front[-1].time:
                        # check that model is not equivalent to another one present already in the pareto front
                        cell_repr = cell_pruning.CellEncoding(model_est.model_encoding)
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
            else:
                # just a rename to integrate with existent code below, it's not a pareto front in this case!
                pareto_front = model_estimations

            # account for case where there are fewer children than K
            children_count = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))

            if not self.pnas_mode:
                # take only the K highest scoring children for next iteration
                children = list(map(lambda child: child.model_encoding, pareto_front[:children_count]))
            else:
                # remove equivalent models, not done already if running in pnas mode
                models = list(map(lambda est: est.model_encoding, pareto_front))
                children, pruned_count = cell_pruning.prune_equivalent_cell_models(models, children_count)
                self._logger.info('Pruned %d equivalent models while selecting CNN children', pruned_count)

            with open(log_service.build_path('csv', 'children.csv'), mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(children)

            # save these children for next round
            self.state_space.update_children(children)
        else:
            self._logger.info("No more updates necessary as max B has been reached!")


class ModelEstimate:
    '''
    Helper class, basically a struct with a function to convert into array for csv saving
    '''

    def __init__(self, model_encoding, score, time):
        self.model_encoding = model_encoding
        self.score = score
        self.time = time

    def to_csv_array(self):
        cell_structure = f"[{';'.join(map(lambda el: str(el), self.model_encoding))}]"
        return [self.time, self.score, cell_structure]

    @staticmethod
    def get_csv_headers():
        return ['time', 'val accuracy', 'cell structure']
