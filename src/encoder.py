from typing import Any, Callable
import log_service
from utils.rstr import rstr
from utils.func_utils import list_flatten


class StateSpace:
    '''
    State Space manager

    Provides utility functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''

    def __init__(self, B, operators,
                 input_lookback_depth=-1,
                 input_lookforward_depth=None):
        '''
        Constructs a search space which models the NAS and PNAS papers

        A single block consists of the 4-tuple:
        (input 1, operation 1, input 2, operation 2)

        The merge operation can be a sum or a concat as required.

        The input operations are used for adding up intermediate values
        inside the same cell. See the NASNet and P-NASNet models to see
        how intermediate blocks connect based on input values.

        The default operation values are based on the P-NAS paper. They
        should be changed as needed.

        # Note:
        This only provides a convenient mechanism to train the networks.
        It is upto the model designer to interpret this block tuple
        information and construct the final model.

        # Args:
            B: Maximum number of blocks

            operators: a list of operations (can be anything, must be
                interpreted by the model designer when constructing the
                actual model. Usually a list of strings.

            input_lookback_depth: should be a negative number.
                Describes how many cells the input should look behind.
                The negative number describes how many cells to look back.
                -1 indicates the last cell (or input image at start), and so on.          

            input_lookforward_depth: sets a limit on input depth that can be looked forward.
                This is useful for scenarios where "flat" models are preferred,
                wherein each cell is flat, though it may take input from deeper
                layers (if the designer so chooses)

                The default searches over cells which can have inter-connections.
                Setting it to 0 limits this to just the current input for that cell (flat cells).
        '''
        self._logger = log_service.get_logger(__name__)

        self.children = None
        self.intermediate_children = None
        
        self.input_encoders = {}    # type: dict[str, Encoder]
        self.operator_encoders = {} # type: dict[str, Encoder]

        self.B = B
        self.input_lookback_depth = input_lookback_depth
        self.input_lookforward_depth = input_lookforward_depth
        assert self.input_lookback_depth < 0, "Invalid lookback_depth value"

        # original values for both inputs and operators
        # since internal block inputs are 0-indexed, B-1 is the last block and therefore not a valid input (excluded)
        self.input_values = list(range(input_lookback_depth, self.B-1))
        if operators is None:
            self.operator_values = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv',
                              '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        else:
            self.operator_values = operators         
        
        # for embedding (see LSTM controller)
        self.inputs_embedding_max = len(self.input_values)
        self.operator_embedding_max = len(self.operator_values)

        # generate categorical encoders for both inputs and operators
        self.input_encoders['cat'] = Encoder('cat', values=self.input_values)
        self.operator_encoders['cat'] = Encoder('cat', values=self.operator_values)

        self.prepare_initial_children()

    def add_input_encoder(self, name: str, fn: Callable[[int], Any]):
        assert fn is not None
        self.input_encoders[name] = Encoder(name, values=self.input_values, fn=fn)

    def add_operator_encoder(self, name: str, fn: Callable[[str], Any]):
        assert fn is not None
        self.operator_encoders[name] = Encoder(name, values=self.operator_values, fn=fn)

    def decode_cell_spec(self, encoded_cell, input_enc_name='cat', op_enc_name='cat'):
        '''
        Parses a list of encoded states to retrieve a list of state values

        # Args:
            state_list: list of encoded states

        # Returns:
            list of state values
        '''
        assert input_enc_name in self.input_encoders.keys() and op_enc_name in self.operator_encoders.keys()
        decode_i = self.input_encoders[input_enc_name].decode
        decode_o = self.operator_encoders[op_enc_name].decode

        return [(decode_i(in1), decode_o(op1), decode_i(in2), decode_o(op2)) for in1, op1, in2, op2 in encoded_cell]

    def encode_cell_spec(self, cell_spec, input_enc_name='cat', op_enc_name='cat', flatten=True):
        '''
        Perform encoding for all blocks in a cell

        # Args:
            child: a list of blocks (which forms one cell / layer)

        # Returns:
            list of entity encoded blocks of the cell
        '''
        assert input_enc_name in self.input_encoders.keys() and op_enc_name in self.operator_encoders.keys()
        encode_i = self.input_encoders[input_enc_name].encode
        encode_o = self.operator_encoders[op_enc_name].encode

        encoded_cell = [(encode_i(in1), encode_o(op1), encode_i(in2), encode_o(op2)) for in1, op1, in2, op2 in cell_spec]
        return list_flatten(encoded_cell) if flatten else encoded_cell

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
        self._logger.info("Search space size : %d", (len(inputs) * (len(self.operator_values) ** 2)))

        search_space = (inputs, self.operator_values)
        self.children = list(self.__construct_permutations(search_space))

    # TODO: not necessary?
    def get_current_step_total_models(self, new_b):
        new_b_dash = new_b - 1 if self.input_lookforward_depth is None \
            else min(self.input_lookforward_depth, new_b)

        possible_input_values = new_b_dash - self.input_lookback_depth

        total_new_child_count = (possible_input_values ** 2) * (len(self.operator_values) ** 2)
        symmetric_child_count = possible_input_values * len(self.operator_values)
        non_specular_child_count = (total_new_child_count + symmetric_child_count) / 2

        return len(self.children) * non_specular_child_count

    def prepare_intermediate_children(self, new_b):
        '''
        Generates the intermediate product of the previous children
        and the current generation of children.

        # Args:
            new_b: the number of blocks in current stage

        # Returns:
            A function that returns a generator that produces a joint of
            the previous and current child models
        '''

        new_b_dash = new_b - 1 if self.input_lookforward_depth is None \
            else min(self.input_lookforward_depth, new_b)

        allowed_input_values = list(range(self.input_lookback_depth, new_b_dash))

        total_new_child_count = (len(allowed_input_values) ** 2) * (len(self.operator_values) ** 2)
        symmetric_child_count = len(allowed_input_values) * len(self.operator_values)
        non_specular_child_count = (total_new_child_count + symmetric_child_count) / 2

        self._logger.info("Obtaining search space for b = %d", new_b)
        self._logger.info("Search space size: %d", non_specular_child_count)

        self._logger.info("Total possible models (considering also equivalent cells): %d", len(self.children) * non_specular_child_count)

        search_space = (allowed_input_values, self.operator_values)
        new_search_space = list(self.__construct_permutations(search_space))

        def generate_models():
            '''
            The generator produce also models with equivalent cells.
            '''
            for child in self.children:
                for permutation in new_search_space:
                    yield child + permutation   # list concat
 
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
                    if in2 >= in1: # added to avoid repeated permutations (equivalent blocks)
                        for op2 in ops:
                            if in2 != in1 or op2 >= op1: # added to avoid repeated permutations (equivalent blocks)
                                yield [(in1, op_enc.decode(op1), in2, op_enc.decode(op2))]


    def print_state_space(self):
        ''' Pretty print the state space '''
        self._logger.info('%s', '*' * 30 + 'STATE SPACE' + '*' * 30)
        self._logger.info('Block values: %s', rstr(list(range(1, self.B + 1))))
        self._logger.info('Inputs: %s', rstr(self.input_values))
        self._logger.info('Operators: %s', rstr(self.operator_values))
        self._logger.info('%s', '*' * 71)

    def print_cell_spec(self, cell_spec):
        ''' Print the cell specification space properly '''
        self._logger.info('Cell specification:')

        # each block is a tuple of 4 elements
        for i, block in enumerate(cell_spec):
            self._logger.info("Block %d: %s", i + 1, rstr(block))

    def update_children(self, children):
        self.children = children

    def get_cells_to_train(self):
        return self.children


class Encoder:
    def __init__(self, name, values: list, fn: Callable=None) -> None:
        self.name = name
        self.values = values
        self.encodings = []

        # dictionary (map) for convertion encoding->value
        self.__index_map = {}

        # dictionary (map) for convertion value->encoding
        # Inverse mapping compared to index_map (see above).
        self.__value_map = {}

        # if encoding function is not provided, use categorical
        for i, val in enumerate(values):
            val_encoding = i+1 if fn is None else fn(val)
            self.__index_map[val_encoding] = val
            self.__value_map[val] = val_encoding
            self.encodings.append(val_encoding)

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
