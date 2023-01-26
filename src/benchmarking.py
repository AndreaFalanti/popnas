import nats_bench

import log_service
from models.results import ClassificationTrainingResults
from search_space import CellSpecification
from utils.rstr import rstr


class NATSbench:
    def __init__(self, bench_path: str) -> None:
        '''
        Initialize NATS-bench API from file. This class provide a general interface to query NATS-bench topology dataset (NAS-bench-201 basically),
        plus some utilities to adapt POPNAS architectures to the benchmark.

        Args:
            bench_path: path to benchmark file (see NATS-bench documentation)
        '''
        super().__init__()

        self.api = nats_bench.create(bench_path, 'tss', fast_mode=True, verbose=False)
        self._logger = log_service.get_logger(__name__)

        # from POPNAS ops (key) to NAS-Bench-201 ops (value).
        # this makes possible to query architectures from NAS-Bench-201 without altering op identifiers chosen for POPNAS.
        # zeroize (none) not present since it is mapped as no connection, without allocating a real operator.
        self.op_conversion_map = {
            '3x3 conv': 'nor_conv_3x3',
            '1x1 conv': 'nor_conv_1x1',
            'identity': 'skip_connect',
            '3x3 avgpool': 'avg_pool_3x3'
        }

    def map_cell_spec_to_nas_bench_201(self, cell_spec: CellSpecification):
        # empty cell case, make it a sort of identity cell
        # it's not the same of POPNAS, since here it would be a network with stem and residuals as reduction cells,
        # still it encapsulates common architectural elements and should be valuable for predictors.
        if cell_spec.is_empty_cell():
            return f'|skip_connect~0|+|none~0|none~1|+|none~0|skip_connect~1|none~2|'

        cell_blocks = len(cell_spec)
        cell_inputs = cell_spec.inputs()
        nas_bench_ops = [self.op_conversion_map[op] for op in cell_spec.operators()]

        # node 0 is -1 input lookback. No need to check inputs for first block, since it is fixed.
        # node 2 is the output of the first block
        op01 = 'skip_connect'   # fixed for our block search space
        op02 = nas_bench_ops[1]      # block 1, operator 2
        op12 = nas_bench_ops[0]      # block 1, operator 1

        # zeroize other edges, except for output of block 1 which is set to skip (otherwise it would be masked in their implementation)
        if cell_blocks == 1:
            op03 = 'none'
            op13 = 'none'
            op23 = 'skip_connect'
        # cell has 2 blocks
        else:
            op03 = nas_bench_ops[2]  # block 2, operator 1 (it is forced to be input -1, since we can't have two edges between the same pair of nodes)
            op13 = nas_bench_ops[3] if cell_inputs[3] == -1 else 'none'
            op23 = nas_bench_ops[3] if cell_inputs[3] == 0 else 'none'

        return f'|{op01}~0|+|{op02}~0|{op12}~1|+|{op03}~0|{op13}~1|{op23}~2|'

    def simulate_training_on_nas_bench_201(self, cell_spec: CellSpecification, dataset_name: str, hp: str = 12):
        architecture_spec = self.map_cell_spec_to_nas_bench_201(cell_spec)
        arch_index = self.api.query_index_by_arch(architecture_spec)
        self._logger.info('The sampled architecture is: %s -> %s', rstr(cell_spec), architecture_spec)
        self._logger.info('The architecture-index is: %d', arch_index)

        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(arch_index, dataset_name, hp=hp)
        # get architecture info about flops and params
        cost_info = self.api.get_cost_info(arch_index, dataset_name, hp=hp)

        # convert accuracy value to [0, 1] interval (probability instead of percentage)
        validation_accuracy /= 100

        self._logger.info('Cumulative current GPU seconds dedicated to NN training: %0.4f', current_total_time_cost)

        return ClassificationTrainingResults(cell_spec, time_cost, latency, cost_info['params'], cost_info['flops'], validation_accuracy, 0)

    def simulate_testing_on_nas_bench_201(self, cell_spec: CellSpecification, dataset_name: str):
        architecture_spec = self.map_cell_spec_to_nas_bench_201(cell_spec)
        arch_index = self.api.query_index_by_arch(architecture_spec)
        self._logger.info('The sampled architecture is: %s -> %s', rstr(cell_spec), architecture_spec)
        self._logger.info('The architecture-index is: %d', arch_index)

        xinfo = self.api.get_more_info(arch_index, dataset=dataset_name, hp="200", is_random=False)
        return xinfo["test-accuracy"] / 100
