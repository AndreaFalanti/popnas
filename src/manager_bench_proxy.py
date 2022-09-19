import logging

import tensorflow as tf

import benchmarking
import log_service
from utils.nn_utils import TrainingResults

# disable Tensorflow info and warning messages (Warning are not on important things, they were investigated. Still, enable them
# when performing changes to see if there are new potential warnings that can affect negatively the algorithm).
tf.get_logger().setLevel(logging.ERROR)


class NetworkBenchManager:
    '''
    Proxy class to get training results of an architecture by querying NAS-bench-201.
    It keeps the same public interface of NetworkManager.
    '''

    def __init__(self, dataset_config: dict):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.
        It also preprocess the dataset, based on the run configuration.
        '''
        self._logger = log_service.get_logger(__name__)

        # self.epochs = cnn_config['epochs']
        self.dataset_name = dataset_config['name']

        # avoids problems with Controller, even if not used
        self.graph_gen = None

        bench_path = dataset_config['path']
        self.nas_bench = benchmarking.NATSbench(bench_path)

    def bootstrap_dataset_lazy_initialization(self):
        '''
        Train the empty cell model for a single epoch, just to load, process and cache the dataset, so that the first model trained in the session
        is not affected by a time estimation bias (which can be very large for datasets generated from image folders).
        '''
        pass

    def perform_proxy_training(self, cell_spec: 'list[tuple]', save_best_model: bool = False):
        '''
        Generate a neural network from the cell specification and trains it for a short amount of epochs to get an estimate
        of its quality. Other relevant metrics of the NN architecture, like the params and flops, are returned together with the training results.

        Args:
            cell_spec (list[tuple]): plain cell specification. Used to build the CNN.
            save_best_model (bool, optional): [description]. Defaults to False.

        Returns:
            (TrainingResults): (reward, timer, total_params, flops, inference_time) of trained network
        '''

        # convert cell spec to NAS-Bench-201 specifications
        # then directly produce the TrainingResults
        return self.nas_bench.simulate_training_on_nas_bench_201(cell_spec, self.dataset_name, hp='200')
