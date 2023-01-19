import itertools
import logging
import re
from typing import Any, Callable, Generator, Sequence, NamedTuple, Iterable

import log_service
from utils.func_utils import list_flatten
from utils.rstr import rstr


class BlockSpecification(NamedTuple):
    '''
    Utility class for specifying blocks in a standard format.

    In plain blocks (no encoding), inputs have *int* type, and operators have *str* type.
    When encoded, they can assume any type.
    '''
    in1: Any
    op1: Any
    in2: Any
    op2: Any

    @staticmethod
    def from_str(block_str: str):
        in1, op1, in2, op2 = block_str.split(', ')
        return BlockSpecification(int(in1), op1, int(in2), op2)

    def __str__(self) -> str:
        return f"({', '.join(map(str, self))})"


class CellSpecification:
    '''
    Utility class for specifying cells in a standard format.
    It basically wraps a list of BlockSpecification and exposes some python dunder methods to be used
    like a Sequence[BlockSpecification], plus other utilities.
    '''
    def __init__(self, data: Iterable[BlockSpecification] = None) -> None:
        self._data = [] if data is None else list(data)
        self._flat_data = list_flatten(self._data)

    def __getitem__(self, index: int) -> BlockSpecification:
        return self._data.__getitem__(index)

    def __len__(self) -> int:
        # avoids potential fictitious blocks (None, None, None, None)
        # TODO: i think fictitious blocks are associate to functions not important for the algorithm, maybe it's possible to get rid of them.
        return len([block for block in self._data if block != (None, None, None, None)])

    def __add__(self, x: 'list[BlockSpecification]') -> 'CellSpecification':
        return CellSpecification(self._data + x)

    def __str__(self) -> str:
        return f"[{';'.join(map(str, self))}]"

    def to_flat_list(self) -> list:
        return self._flat_data

    def pretty_logging(self, logger: logging.Logger):
        ''' Print the cell specification space properly '''
        logger.info('Cell specification:')
        for i, block in enumerate(self._data):
            logger.info("\tBlock %d: %s", i + 1, block)

    def inputs(self) -> list:
        return self._flat_data[::2]

    def operators(self) -> list:
        return self._flat_data[1::2]

    @staticmethod
    def from_str(cell_str: str) -> 'CellSpecification':
        # empty cell case
        if cell_str == '[]':
            return CellSpecification()

        cell_str = re.sub(r'[\[\]\'\"()]', '', cell_str)
        return CellSpecification([BlockSpecification.from_str(tuple_str) for tuple_str in cell_str.split(';')])

    def is_empty_cell(self):
        return len(self._data) == 0

    def is_specular_monoblock(self):
        if len(self._data) != 1:
            return False

        block = self._data[0]
        return block.in1 == block.in2 and block.op1 == block.op2 and block.in1 == -1


class SearchSpace:
    '''
    Search space manager for cell-based approaches.

    Provides utility functions for holding cell specifications for each step, that the network manager then use
    for building the actual CNNs.
    It also contains functions for expanding these networks progressively, and the encoders to adapt the cell specifications representation
    for various use cases.
    '''

    def __init__(self, ss_config: 'dict[str, Any]', benchmarking: bool = False):
        '''
        Constructs a search space which models the NASNet and PNAS papers, producing encodings for blocks and cells.
        The encodings are useful to store architecture information, from which the models can be generated.

        A single block encoding consists of the 4-tuple: (input 1, operation 1, input 2, operation 2).
        A cell encoding is instead a list of block encodings.

        The operators are used for adding up intermediate values inside the same cell.
        See the NASNet and PNASNet models to see how intermediate blocks connect based on input values.

        Args:
            ss_config: search space configuration provided in the input configuration file
            benchmarking: set it to True when performing NATS-bench runs
        '''
        self._logger = log_service.get_logger(__name__)
        self.benchmarking = benchmarking

        self.children = []  # type: list[CellSpecification]
        self.exploration_front = []  # type: list[CellSpecification]

        self.input_encoders = {}  # type: dict[str, Encoder]
        self.operator_encoders = {}  # type: dict[str, Encoder]

        self.B = ss_config['blocks']
        self.input_lookback_depth = ss_config['lookback_depth']    # positive value
        self.operator_values = ss_config['operators']

        if self.input_lookback_depth < 0:
            raise ValueError('Invalid lookback_depth value')
        if self.operator_values is None or len(self.operator_values) == 0:
            raise ValueError('No operators have been provided in search space')

        # original values for both inputs and operators
        # since internal block inputs are 0-indexed, B-1 is the last block and therefore not a valid input (excluded)
        self.input_values = list(range(-self.input_lookback_depth, self.B - 1))

        # for embedding (see LSTM controller), use categorical values
        # all values must be strictly < than these (that's the reason for + 1)
        self.inputs_embedding_max = len(self.input_values) + 1
        self.operator_embedding_max = len(self.operator_values) + 1

        # generate categorical encoders for both inputs and operators
        self.input_encoders['cat'] = Encoder('cat', values=self.input_values)
        self.operator_encoders['cat'] = Encoder('cat', values=self.operator_values)

        # print info about search space to the logger
        self.print_search_space()

        self.prepare_initial_children()

    def get_lookback_inputs(self):
        return self.input_values[:self.input_lookback_depth]

    def get_allowed_inputs(self, current_b: int):
        return self.input_values[:self.input_lookback_depth + current_b - 1]

    def add_input_encoder(self, name: str, fn: Callable[[int], Any]):
        assert fn is not None
        self.input_encoders[name] = Encoder(name, values=self.input_values, fn=fn)

    def add_operator_encoder(self, name: str, fn: Callable[[str], Any]):
        assert fn is not None
        self.operator_encoders[name] = Encoder(name, values=self.operator_values, fn=fn)

    def decode_cell_spec(self, encoded_cell: CellSpecification, input_enc_name='cat', op_enc_name='cat') -> CellSpecification:
        '''
        Decodes an encoded cell specification.

        Args:
            encoded_cell: encoded cell specification
            input_enc_name: name of input encoder
            op_enc_name: name of operator encoder

        Returns:
            decoded cell
        '''
        assert input_enc_name in self.input_encoders.keys() and op_enc_name in self.operator_encoders.keys()
        decode_i = self.input_encoders[input_enc_name].decode
        decode_o = self.operator_encoders[op_enc_name].decode

        return CellSpecification([BlockSpecification(decode_i(in1), decode_o(op1), decode_i(in2), decode_o(op2))
                                  for in1, op1, in2, op2 in encoded_cell])

    def encode_cell_spec(self, cell_spec: CellSpecification, input_enc_name='cat', op_enc_name='cat') -> CellSpecification:
        '''
        Perform encoding for all blocks in a cell.

        Args:
            cell_spec: plain cell specification
            input_enc_name: name of input encoder
            op_enc_name: name of operator encoder
            flatten: if True, a flat list is returned instead of a list of tuples

        Returns:
            encoded cell
        '''
        assert input_enc_name in self.input_encoders.keys() and op_enc_name in self.operator_encoders.keys()
        encode_i = self.input_encoders[input_enc_name].encode
        encode_o = self.operator_encoders[op_enc_name].encode

        return CellSpecification([BlockSpecification(encode_i(in1), encode_o(op1), encode_i(in2), encode_o(op2))
                                  for in1, op1, in2, op2 in cell_spec])

    def __compute_non_specular_expansions_count(self, valid_inputs: list):
        '''
        Returns the count of all possible expansions, given the allowed input values for the step.
        To get the total CNN children for steps B >= 2, multiply this value for the total CNN trained at previous step.
        Args:
            valid_inputs (list): allowed input values

        Returns:
            (int): expansions count (total CNN to train for step B = 1)
        '''
        total_expansions_count = (len(valid_inputs) ** 2) * (len(self.operator_values) ** 2)
        symmetric_expansions_count = len(valid_inputs) * len(self.operator_values)
        # non symmetric expansions count is returned
        return (total_expansions_count + symmetric_expansions_count) / 2

    def prepare_initial_children(self):
        '''
        Prepare the initial set of cells. All these cells must be trained to obtain the initial set of scores to boostrap the predictors.
        Next cells will be built upon these cell specifications.
        '''
        # Take only first elements that refers to lookback values, which are the only allowed values initially.
        lb_inputs = self.get_lookback_inputs()

        # if input_lookback_depth == 0, then we need to adjust to have at least one input (generally -1, previous cell)
        if len(lb_inputs) == 0:
            lb_inputs = [-1]

        self._logger.info("Obtaining search space for b = 1")
        self._logger.info("Search space size : %d", self.__compute_non_specular_expansions_count(lb_inputs))

        self.children = [CellSpecification([block]) for block in self.__construct_new_blocks_permutations(lb_inputs)]

    def perform_children_expansion(self, curr_b: int, use_exploration_front: bool = True):
        '''
        Expand the children with an additional block.

        Args:
            curr_b: the number of blocks in current stage
            use_exploration_front: use exploration front networks as baseline for progressive expansion

        Returns:
            A function that returns a generator for producing iteratively the cell expansions
        '''

        if use_exploration_front:
            self.children = self.children + self.exploration_front

        # last block index can't be used by any block, so -1 to exclude it
        allowed_input_values = self.get_allowed_inputs(curr_b)
        non_specular_expansions = self.__compute_non_specular_expansions_count(allowed_input_values)

        self._logger.info("Obtaining search space for b = %d", curr_b)
        self._logger.info("Search space size: %d", non_specular_expansions)
        self._logger.info("Total possible models (considering also equivalent cells): %d", len(self.children) * non_specular_expansions)

        new_blocks = list(self.__construct_new_blocks_permutations(allowed_input_values))

        # NAS-bench-201 supports two blocks, but second block can't have both inputs as 0 (first block output), since it is not
        # supported by NAS-Bench-201 network structure.
        if self.benchmarking:
            new_blocks = [block for block in new_blocks for in1, _, in2, _ in block if not (in1 == 0 and in2 == 0)]

        def generate_models() -> Generator[CellSpecification, None, None]:
            '''
            The generator produces also models with equivalent cells.
            '''
            for child in self.children:
                for block in new_blocks:
                    yield child + [block]  # list concat

        return generate_models

    def __construct_new_blocks_permutations(self, allowed_inputs: Sequence) -> Generator[BlockSpecification, None, None]:
        '''
        Build a generator which returns all blocks which can be built from provided inputs and operators.
        Equivalent blocks (example: [-2, A, -1, A] and [-1, A, -2, A]) are excluded from the search space.
        '''
        # Use int categorical encodings for operators to allow easier specular blocks check.
        # They are reconverted in actual values (strings) when returned.
        op_enc = self.operator_encoders['cat']
        ops = op_enc.encodings

        for in1 in allowed_inputs:
            for op1 in ops:
                for in2 in allowed_inputs:
                    if in2 >= in1:  # added to avoid repeated permutations (equivalent blocks)
                        for op2 in ops:
                            if in2 != in1 or op2 >= op1:  # added to avoid repeated permutations (equivalent blocks)
                                yield BlockSpecification(in1, op_enc.decode(op1), in2, op_enc.decode(op2))

    def generate_eqv_cells(self, cell_spec: CellSpecification, size: int = None):
        # that is basically the 'fixed blocks' index list
        used_block_outputs = set(filter(lambda el: el >= 0, cell_spec.inputs()))

        # a block is swappable (can change position inside the cell) if its output is not used by other blocks
        swappable_blocks_mask = [(i not in used_block_outputs) for i in range(len(cell_spec))]
        swappable_blocks = [block for block, flag in zip(cell_spec, swappable_blocks_mask) if flag]

        # add NULL blocks (all None) to swappable blocks, to reach the given cell size
        if size is not None:
            assert size - len(cell_spec) >= 0
            for _ in range(size - len(cell_spec)):
                swappable_blocks.append(BlockSpecification(None, None, None, None))

        # generate all possible permutations of the swappable blocks
        eqv_swap_only_set = set(itertools.permutations(swappable_blocks))

        eqv_cells = []
        for cell in eqv_swap_only_set:
            # cell is a tuple containing the block tuples; it must be converted into a list of tuples
            cell = [*cell]

            # add the non-swappable blocks in their correct positions
            for block_index in used_block_outputs:
                cell.insert(block_index, cell_spec[block_index])
            eqv_cells.append(CellSpecification(cell))

        return eqv_cells, used_block_outputs

    def print_search_space(self):
        ''' Pretty print the state space '''
        self._logger.info('%s', '*' * 43 + ' SEARCH SPACE ' + '*' * 43)
        self._logger.info('Block values: %s', rstr(list(range(1, self.B + 1))))
        self._logger.info('Inputs: %s', rstr(self.input_values))
        self._logger.info('Operators: %s', rstr(self.operator_values))
        self._logger.info('%s', '*' * 100)


class Encoder:
    def __init__(self, name: str, values: list, fn: Callable = None, none_val=0) -> None:
        self.name = name
        self.values = values
        self.encodings = []

        # dictionary (map) for conversion encoding->value
        self.__index_map = {}

        # dictionary (map) for conversion value->encoding
        # Inverse mapping compared to index_map (see above).
        self.__value_map = {}

        for i, val in enumerate(values):
            # if encoding function is not provided, use categorical
            encoding = i + 1 if fn is None else fn(val)

            self.__index_map[encoding] = val
            self.__value_map[val] = encoding
            self.encodings.append(encoding)

        # for generate_eqv_cells purposes, an encoding to None must be provided.
        self.__value_map[None] = none_val
        self.__index_map[none_val] = None

    def return_metadata(self):
        return {
            'name': self.name,
            'values': self.values,
            'encodings': self.encodings,
            'size': len(self.values),
            'index_map': self.__index_map,
            'value_map': self.__value_map,
        }

    def encode(self, value):
        '''
        Embed value to categorical
        '''
        return self.__value_map[value]

    def decode(self, index):
        '''
        Decode categorical to original value
        '''
        return self.__index_map[index]


def parse_cell_strings(cell_structures: Iterable[str]) -> 'list[CellSpecification]':
    '''
    Function used to parse in an actual python structure the csv field storing the non-encoded cell structure, which is saved in form:
    "[(in1,op1,in2,op2);(...);...]"
    '''
    return [CellSpecification.from_str(cell_str) for cell_str in cell_structures]
