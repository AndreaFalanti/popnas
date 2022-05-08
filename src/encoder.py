import itertools
from typing import Any, Callable

import log_service
from utils.func_utils import list_flatten
from utils.rstr import rstr


class SearchSpace:
    '''
    Search space manager for cell-based approaches.

    Provides utility functions for holding cell specifications for each step, that the network manager then use
    for building the actual CNNs.
    It also contains functions for expanding these network progressively and the encoders to adapt the cell specifications representation
    for various use cases.
    '''

    def __init__(self, B: int, operators: 'list[str]', cell_stack_depth: int,
                 input_lookback_depth: int = -1, input_lookforward_depth: int = None):
        '''
        Constructs a search space which models the NASNet and PNAS papers, producing encodings for blocks and cells.
        The encodings are useful to store architecture information, from which the models can be generated.

        A single block encoding consists of the 4-tuple: (input 1, operation 1, input 2, operation 2).
        A cell encoding is instead a list of block encodings.

        The operators are used for adding up intermediate values
        inside the same cell. See the NASNet and PNASNet models to see
        how intermediate blocks connect based on input values.

        The default operator values are based on the PNAS paper. They
        should be changed as needed.

        Args:
            B: Maximum number of blocks

            operators: a list of operations (can be anything, must be
                interpreted by the model designer when constructing the
                actual model. Usually a list of strings.

            input_lookback_depth: should be a negative number.
                Describes how many cells the input should look behind.
                The negative number describes how many cells to look back.
                -1 indicates the last cell (or input image at start), and so on.          

            input_lookforward_depth: (TODO: not supported) sets a limit on input depth that can be looked forward.
                This is useful for scenarios where "flat" models are preferred,
                wherein each cell is flat, though it may take input from deeper
                layers (if the designer so chooses)

                The default searches over cells which can have inter-connections.
                Setting it to 0 limits this to just the current input for that cell (flat cells).
        '''
        self._logger = log_service.get_logger(__name__)

        self.children = []
        self.intermediate_children = []
        self.exploration_front = []

        self.input_encoders = {}  # type: dict[str, Encoder]
        self.operator_encoders = {}  # type: dict[str, Encoder]

        if input_lookback_depth >= 0:
            raise ValueError('Invalid lookback_depth value')
        if input_lookforward_depth is not None:
            raise NotImplementedError('Lookforward inputs are actually not supported')

        self.B = B
        self.input_lookback_depth = input_lookback_depth
        self.input_lookforward_depth = input_lookforward_depth

        self.cell_stack_depth = cell_stack_depth

        # original values for both inputs and operators
        # since internal block inputs are 0-indexed, B-1 is the last block and therefore not a valid input (excluded)
        self.input_values = list(range(input_lookback_depth, self.B - 1))
        if operators is None or len(operators) == 0:
            self.operator_values = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv',
                                    '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        else:
            self.operator_values = operators

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

    def add_input_encoder(self, name: str, fn: Callable[[int], Any]):
        assert fn is not None
        self.input_encoders[name] = Encoder(name, values=self.input_values, fn=fn)

    def add_operator_encoder(self, name: str, fn: Callable[[str], Any]):
        assert fn is not None
        self.operator_encoders[name] = Encoder(name, values=self.operator_values, fn=fn)

    def decode_cell_spec(self, encoded_cell, input_enc_name='cat', op_enc_name='cat'):
        '''
        Decodes an encoded cell specification.

        Args:
            encoded_cell (list): encoded cell specification
            input_enc_name: name of input encoder
            op_enc_name: name of operator encoder

        Returns:
            (list): decoded cell
        '''
        assert input_enc_name in self.input_encoders.keys() and op_enc_name in self.operator_encoders.keys()
        decode_i = self.input_encoders[input_enc_name].decode
        decode_o = self.operator_encoders[op_enc_name].decode

        return [(decode_i(in1), decode_o(op1), decode_i(in2), decode_o(op2)) for in1, op1, in2, op2 in encoded_cell]

    def encode_cell_spec(self, cell_spec, input_enc_name='cat', op_enc_name='cat', flatten=True):
        '''
        Perform encoding for all blocks in a cell

        Args:
            cell_spec (list): plain cell specification
            input_enc_name: name of input encoder
            op_enc_name: name of operator encoder
            flatten (bool): if True, a flat list is returned instead of a list of tuples

        Returns:
            (list): encoded cell
        '''
        assert input_enc_name in self.input_encoders.keys() and op_enc_name in self.operator_encoders.keys()
        encode_i = self.input_encoders[input_enc_name].encode
        encode_o = self.operator_encoders[op_enc_name].encode

        encoded_cell = [(encode_i(in1), encode_o(op1), encode_i(in2), encode_o(op2)) for in1, op1, in2, op2 in cell_spec]
        return list_flatten(encoded_cell) if flatten else encoded_cell

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
        Prepare the initial set of child models which must
        all be trained to obtain the initial set of scores
        '''
        # Take only first elements that refers to lookback values, which are the only allowed values initially.
        inputs = self.input_values[:abs(self.input_lookback_depth)]

        # if input_lookback_depth == 0, then we need to adjust to have at least one input (generally -1, previous cell)
        if len(inputs) == 0:
            inputs = [-1]

        self._logger.info("Obtaining search space for b = 1")
        self._logger.info("Search space size : %d", self.__compute_non_specular_expansions_count(inputs))

        search_space = (inputs, self.operator_values)
        self.children = list(self.__construct_permutations(search_space))

    def prepare_intermediate_children(self, new_b, use_exploration_front: bool = True):
        '''
        Generates the intermediate product of the previous children
        and the current generation of children.

        Args:
            new_b: the number of blocks in current stage
            use_exploration_front: use exploration front networks as baseline for progressive expansion

        Returns:
            A function that returns a generator that produces a joint of
            the previous and current child models
        '''

        if use_exploration_front:
            self.children = self.children + self.exploration_front

        new_b_dash = new_b - 1 if self.input_lookforward_depth is None \
            else min(self.input_lookforward_depth, new_b)

        allowed_input_values = list(range(self.input_lookback_depth, new_b_dash))
        non_specular_expansions = self.__compute_non_specular_expansions_count(allowed_input_values)

        self._logger.info("Obtaining search space for b = %d", new_b)
        self._logger.info("Search space size: %d", non_specular_expansions)

        self._logger.info("Total possible models (considering also equivalent cells): %d", len(self.children) * non_specular_expansions)

        search_space = (allowed_input_values, self.operator_values)
        new_search_space = list(self.__construct_permutations(search_space))

        def generate_models():
            '''
            The generator produce also models with equivalent cells.
            '''
            for child in self.children:
                for permutation in new_search_space:
                    yield child + permutation  # list concat

        return generate_models

    def __construct_permutations(self, search_space):
        '''
        State space is a 2-tuple (inputs set, operators set).
        Equivalent blocks (example: [-2, A, -1, A] and [-1, A, -2, A]) are excluded from the search space.
        '''
        inputs, ops = search_space

        # Use int categorical encodings for operators to allow easier specular blocks check.
        # They are reconverted in actual values (strings) when returned.
        op_enc = self.operator_encoders['cat']
        ops = op_enc.encodings

        for in1 in inputs:
            for op1 in ops:
                for in2 in inputs:
                    if in2 >= in1:  # added to avoid repeated permutations (equivalent blocks)
                        for op2 in ops:
                            if in2 != in1 or op2 >= op1:  # added to avoid repeated permutations (equivalent blocks)
                                yield [(in1, op_enc.decode(op1), in2, op_enc.decode(op2))]

    def generate_eqv_cells(self, cell_spec: list, size: int = None):
        cell_inputs = list_flatten(cell_spec)[::2]
        # that is basically the 'fixed blocks' index list
        used_block_outputs = set(filter(lambda el: el >= 0, cell_inputs))

        # a block is swappable (can change position inside the cell) if its output is not used by other blocks
        swappable_blocks_mask = [(i not in used_block_outputs) for i in range(len(cell_spec))]
        swappable_blocks = [block for block, flag in zip(cell_spec, swappable_blocks_mask) if flag]

        # add NULL blocks (all None) to swappable blocks, to reach the given cell size
        if size is not None:
            assert size - len(cell_spec) >= 0
            for _ in range(size - len(cell_spec)):
                swappable_blocks.append((None, None, None, None))

        # generate all possible permutations of the swappable blocks
        eqv_swap_only_set = set(itertools.permutations(swappable_blocks))

        eqv_cells = []
        for cell in eqv_swap_only_set:
            # cell is a tuple containing the block tuples, it must be converted into a list of tuples
            cell = [*cell]

            # add the non-swappable blocks in their correct positions
            for block_index in used_block_outputs:
                cell.insert(block_index, cell_spec[block_index])
            eqv_cells.append(cell)

        return eqv_cells, used_block_outputs

    def print_search_space(self):
        ''' Pretty print the state space '''
        self._logger.info('%s', '*' * 43 + ' SEARCH SPACE ' + '*' * 43)
        self._logger.info('Block values: %s', rstr(list(range(1, self.B + 1))))
        self._logger.info('Inputs: %s', rstr(self.input_values))
        self._logger.info('Operators: %s', rstr(self.operator_values))
        self._logger.info('Total cells stacked in each CNN: %d', self.cell_stack_depth)
        self._logger.info('%s', '*' * 100)

    def print_cell_spec(self, cell_spec):
        ''' Print the cell specification space properly '''
        self._logger.info('Cell specification:')

        # each block is a tuple of 4 elements
        for i, block in enumerate(cell_spec):
            self._logger.info("Block %d: %s", i + 1, rstr(block))


class Encoder:
    def __init__(self, name, values: list, fn: Callable = None, none_val=0) -> None:
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
