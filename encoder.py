import numpy as np
import pprint
from collections import OrderedDict

import os

import log_service
_logger = log_service.getLogger(__name__)


if not os.path.exists('logs/'):
    os.makedirs('logs/')

# TODO: deleting this would cause import failures inside aMLLibrary files, but from POPNAS its better to 
# import them directly to enable intellisense
import sys
sys.path.insert(1, 'aMLLibrary')

class StateSpace:
    '''
    State Space manager

    Provides utility functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self, B, operators,
                 input_lookback_depth=0,
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

            input_lookback_depth: should be a negative number or 0.
                Describes how many cells the input should look behind.
                Can be used to tensor information from 0 or more cells from
                the current cell.

                The negative number describes how many cells to look back.
                Set to 0 if the lookback feature is not needed (flat cells).

            input_lookforward_depth: sets a limit on input depth that can be looked forward.
                This is useful for scenarios where "flat" models are preferred,
                wherein each cell is flat, though it may take input from deeper
                layers (if the designer so chooses)

                The default searches over cells which can have inter-connections.
                Setting it to 0 limits this to just the current input for that cell (flat cells).
        '''
        self.states = OrderedDict()
        self.state_count_ = 0
        
        self.children = None
        self.intermediate_children = None

        self.B = B

        if operators is None:
            self.operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv',
                              '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        else:
            self.operators = operators

        self.input_lookback_depth = input_lookback_depth
        self.input_lookforward_depth = input_lookforward_depth

        input_values = list(range(input_lookback_depth, self.B-1))  # -1 = Hc-1, 0-(B-1) = Hci
        self.inputs_embedding_max = len(input_values)
        # self.operator_embedding_max = len(np.unique(operators))
        self.operator_embedding_max = len(np.unique(self.operators))

        self._add_state('inputs', values=input_values)
        self._add_state('ops', values=self.operators)
        self.prepare_initial_children()

    def _add_state(self, name, values):
        '''
        Adds a "state" to the state manager, along with some metadata for efficient
        packing and unpacking of information required by the RNN ControllerManager.

        Stores metadata such as:
        -   Global ID
        -   Name
        -   Valid Values
        -   Number of valid values possible
        -   Map from value ID to state value
        -   Map from state value to value ID

        # Args:
            name: name of the state / action
            values: valid values that this state can take

        # Returns:
            Global ID of the state. Can be used to refer to this state later.
        '''
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def embedding_encode(self, id, value):
        '''
        Embedding index encode the specific state value

        # Args:
            id: global id of the state
            value: state value

        # Returns:
            embedding encoded representation of the state value
        '''
        state = self[id]
        value_map = state['value_map_']
        value_idx = value_map[value]
        encoding = np.zeros((1, 1), dtype=np.float32)
        encoding[0, 0] = value_idx

        return encoding

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        # Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        # Returns:
            The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']
        value = index_map[index]

        return value

    def parse_state_space_list(self, state_list):
        '''
        Parses a list of one hot encoded states to retrieve a list of state values

        # Args:
            state_list: list of one hot encoded states

        # Returns:
            list of state values
        '''
        state_values = []
        for id, state_value in enumerate(state_list):
            state_val_idx = state_value[0, 0]
            value = self.get_state_value(id % 2, state_val_idx)
            state_values.append(value)

        return state_values

    def entity_encode_child(self, child):
        '''
        Perform encoding for all blocks in a cell

        # Args:
            child: a list of blocks (which forms one cell / layer)

        # Returns:
            list of entity encoded blocks of the cell
        '''
        encoded_child = []
        for i, val in enumerate(child):
            encoded_child.append(self.embedding_encode(i % 2, val))
        return encoded_child

    def prepare_initial_children(self):
        '''
        Prepare the initial set of child models which must
        all be trained to obtain the initial set of scores
        '''
        # set of all operations
        ops = list(range(len(self.operators)))
        inputs = list(range(self.input_lookback_depth, 0))

        # if input_lookback_depth == 0, then we need to adjust to have at least
        # one input (generally 0)
        if len(inputs) == 0:
            inputs = [0]

        _logger.info("Obtaining search space for b = 1")
        print("Search space size : %d", (len(inputs) * (len(self.operators) ** 2)))

        search_space = [inputs, ops, inputs, ops]
        self.children = list(self._construct_permutations(search_space))

    def prepare_intermediate_children(self, new_b):
        '''
        Generates the intermediate product of the previous children
        and the current generation of children.

        # Args:
            new_b: the number of blocks in current stage

        # Returns:
            a generator that produces a joint of the previous and current
            child models
        '''
        if self.input_lookforward_depth is not None:
            new_b_dash = min(self.input_lookforward_depth, new_b)
        else:
            new_b_dash = new_b - 1
            
        new_ip_values = list(range(self.input_lookback_depth, new_b_dash))
        ops = list(range(len(self.operators)))

        # if input_lookback_depth == 0, then we need to adjust to have at least
        # one input (generally 0)
        if len(new_ip_values) == 0:
            new_ip_values = [0]

        new_child_count = ((len(new_ip_values)) ** 2) * (len(self.operators) ** 2)
        _logger.info("Obtaining search space for b = %d", new_b)
        _logger.info("Search space size: %d", new_child_count)

        _logger.info("Total models to evaluate: %d", (len(self.children) * new_child_count))

        search_space = [new_ip_values, ops, new_ip_values, ops]
        new_search_space = list(self._construct_permutations(search_space))

        for i, child in enumerate(self.children):
            for permutation in new_search_space:
                temp_child = list(child)
                temp_child.extend(permutation)
                yield temp_child

    def _construct_permutations(self, search_space):
        ''' state space is a 4-tuple (ip1, op1, ip2, op2) '''
        for input1 in search_space[0]:
            for operation1 in search_space[1]:
                for input2 in search_space[2]:
                    #if input2 >= input1: # added to avoid repeated permutations
                    for operation2 in search_space[3]:
                            #if (input2 != input1) or operation1 >= operation2: # added to avoid repeated permutations
                            yield (input1, self.operators[operation1], input2, self.operators[operation2])

    def print_state_space(self):
        ''' Pretty print the state space '''
        _logger.info('%s', '*' * 40 + 'STATE SPACE' + '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            _logger.info(pp.pformat(state))

    def print_actions(self, actions):
        ''' Print the action space properly '''
        _logger.info('Actions :')

        for id, action in enumerate(actions):
            state = self[id]
            name = state['name']
            vals = [(self.get_state_value(id % 2, p), p) for n, p in zip(state['values'], *action)]
            _logger.info("%s : %s", name, vals)

    def update_children(self, children):
        self.children = children

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_

    def print_total_models(self, K):
        ''' Compute the total number of models to generate and train '''
        num_inputs = 1 if self.input_lookback_depth == 0 else abs(self.input_lookback_depth)
        level1 = (num_inputs ** 2) * (len(self.operators) ** 2)
        remainder = (self.B - 1) * K
        total = level1 + remainder
        return total
