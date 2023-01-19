import benchmarking
import log_service
from search_space import CellSpecification


class GraphProxy:
    def __init__(self, bench: benchmarking.NATSbench, arch_index: int) -> None:
        super().__init__()
        self.bench = bench
        self.arch_index = arch_index

    def get_total_params(self):
        # always done on cifar10, since parameters do not vary between datasets
        return self.bench.api.get_cost_info(self.arch_index, dataset='cifar10', hp="200")['params']


class GraphGeneratorProxy:
    def __init__(self, bench: benchmarking.NATSbench) -> None:
        super().__init__()
        self.bench = bench

    def generate_network_graph(self, cell_spec: CellSpecification):
        arch = self.bench.map_cell_spec_to_nas_bench_201(cell_spec)
        arch_index = self.bench.api.query_index_by_arch(arch)
        return GraphProxy(self.bench, arch_index)


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

        bench_path = dataset_config['path']
        self.nas_bench = benchmarking.NATSbench(bench_path)

        # make possible for the controller to retrieve params
        self.graph_gen = GraphGeneratorProxy(self.nas_bench)

    def bootstrap_dataset_lazy_initialization(self):
        '''
        Train the empty cell model for a single epoch, just to load, process and cache the dataset, so that the first model trained in the session
        is not affected by a time estimation bias (which can be very large for datasets generated from image folders).
        '''
        pass

    def perform_proxy_training(self, cell_spec: CellSpecification, save_best_model: bool = False):
        '''
        Queries the results of the specified network from the NAS bench, simulating the proxy training.
        '''

        # convert cell spec to NAS-Bench-201 specifications, then directly produce the TrainingResults
        return self.nas_bench.simulate_training_on_nas_bench_201(cell_spec, self.dataset_name, hp='200')
