import numpy as np
import pprint
from collections import OrderedDict

import log_service

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

        assert self.input_lookback_depth < 0, "Invalid lookback_depth value"

        input_values = list(range(input_lookback_depth, self.B-1))  # -1 = Hc-1, 0-(B-1) = Hci
        self.inputs_embedding_max = len(input_values)
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

        self._logger.info("Obtaining search space for b = 1")
        self._logger.info("Search space size : %d", (len(inputs) * (len(self.operators) ** 2)))

        search_space = [inputs, ops, inputs, ops]
        self.children = list(self._construct_permutations(search_space))

    def get_current_step_total_models(self, new_b):
        new_b_dash = new_b - 1 if self.input_lookforward_depth is None \
            else min(self.input_lookforward_depth, new_b)

        possible_input_values = new_b_dash - self.input_lookback_depth

        total_new_child_count = (possible_input_values ** 2) * (len(self.operators) ** 2)
        symmetric_child_count = possible_input_values * len(self.operators)
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

        possible_input_values = list(range(self.input_lookback_depth, new_b_dash))
        ops = list(range(len(self.operators)))

        total_new_child_count = (len(possible_input_values) ** 2) * (len(self.operators) ** 2)
        symmetric_child_count = len(possible_input_values) * len(self.operators)
        non_specular_child_count = (total_new_child_count + symmetric_child_count) / 2

        self._logger.info("Obtaining search space for b = %d", new_b)
        self._logger.info("Search space size: %d", non_specular_child_count)

        self._logger.info("Total possible models (considering also equivalent cells): %d", len(self.children) * non_specular_child_count)

        search_space = [possible_input_values, ops, possible_input_values, ops]
        new_search_space = list(self._construct_permutations(search_space))

        def generate_models():
            '''
            The generator produce also models with equivalent cells.
            '''
            for _, child in enumerate(self.children):
                for permutation in new_search_space:
                    temp_child = list(child)
                    temp_child.extend(permutation)
                    yield temp_child

        def get_all_unique_models():
            '''
            Produce directly all models and return them. It prunes equivalent models and also
            returns the amount of models pruned.
            '''
            children = []
            for _, child in enumerate(self.children):
                for permutation in new_search_space:
                    temp_child = list(child)
                    temp_child.extend(permutation)
                    children.append(temp_child)
            
            return self._prune_equivalent_cell_models(children)
 
        return generate_models, get_all_unique_models

    def _prune_equivalent_cell_models(self, models):
        prime_models = []
        prime_cell_repr = []
        #pruned_models = []  # DEBUG only
        pruned_count = 0

        for model in models:
            # build a better cell representation for equivalence purposes
            cell_repr = CellEncoding(model)

            # check possible equivalence with other models already generated
            if self._check_model_equivalence(cell_repr, prime_cell_repr):
                pruned_count += 1
                #pruned_models.append(model)     # DEBUG only
            else:
                prime_cell_repr.append(cell_repr)
                prime_models.append(model)

        return prime_models, pruned_count

    def _check_model_equivalence(self, model, generated_models):
        for gen_model in generated_models:
            if model == gen_model:
                return True

        return False

    def _construct_permutations(self, search_space):
        '''
        State space is a 4-tuple (ip1, op1, ip2, op2).
        Equivalent blocks (example: [-2, A, -1, A] and [-1, A, -2, A]) are excluded from the search space.
        '''
        for input1 in search_space[0]:
            for operation1 in search_space[1]:
                for input2 in search_space[2]:
                    if input2 >= input1: # added to avoid repeated permutations (equivalent blocks)
                        for operation2 in search_space[3]:
                            if input2 != input1 or operation2 >= operation1: # added to avoid repeated permutations (equivalent blocks)
                                yield (input1, self.operators[operation1], input2, self.operators[operation2])


    def print_state_space(self):
        ''' Pretty print the state space '''
        self._logger.info('%s', '*' * 40 + 'STATE SPACE' + '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            self._logger.info(pp.pformat(state))

        self._logger.info('%s', '*' * 90)

    def print_actions(self, actions):
        ''' Print the action space properly '''
        self._logger.info('Actions:')

        for id, action in enumerate(actions):
            state = self[id]
            name = state['name']
            vals = [(self.get_state_value(id % 2, p), p) for n, p in zip(state['values'], *action)]
            self._logger.info("%s : %s", name, vals)

    def update_children(self, children):
        self.children = children

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_


# These classes are used for representing models in a more structured way, so that it's possible to compare
# them more easily (trim specular cells and equivalent models in general).
# TODO: name could be missleading, they are not actual encodings used by algorithm.

class OpEncoding:
    def __init__(self, input, op) -> None:
        self.input = input  # input can be either a number or a BlockEncoding (if input >= 0, other block output)
        self.op = op

    def __eq__(self, o: object) -> bool:
        if isinstance(o, OpEncoding):
            return self.input == o.input and self.op == o.op
        return False

class BlockEncoding:
    def __init__(self, block_spec) -> None:
        self.L = OpEncoding(block_spec[0], block_spec[1])
        self.R = OpEncoding(block_spec[2], block_spec[3])

    def __eq__(self, o: object) -> bool:
        if isinstance(o, BlockEncoding):
            return (self.L == o.L and self.R == o.R) or \
                (self.L == o.R and self.R == o.L)
        return False

class CellEncoding:
    def __init__(self, model_list: list) -> None:
        self.blocks = []
        blocks_num = len(model_list) // 4

        for i in range(blocks_num):
            block_list = model_list[i*4:(i+1)*4]

            # iterate the two inputs to substitute it in case it's from another block
            inputs = block_list[::2]    # 0 and 2 index
            for j, input in enumerate(inputs):
                # use another block output as input, substitute input index with BlockEncoding
                if input >= 0:
                    block_list[j*2] = self.blocks[input]

            self.blocks.append(BlockEncoding(block_list))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, CellEncoding) and len(self.blocks) == len(o.blocks):
            # each block must be equivalent to a not already eqv matched block for
            # having a complete cell equivalence.
            # The mask helps filtering already matched blocks.
            blocks_mask = [True for _ in o.blocks]

            for block_enc in self.blocks:
                eqv_to_other_cell_block = False
                # index produced by enumerate is required to update the boolean mask
                for i, viable_block_enc in filter(lambda tuple: blocks_mask[tuple[0]], enumerate(o.blocks)):
                    if block_enc == viable_block_enc:
                        # this block is no more viable for further equalities
                        blocks_mask[i] = False
                        eqv_to_other_cell_block = True
                        break

                # if any block is not equivalent to a viable one, the cells are different
                if not eqv_to_other_cell_block:
                    return False

            return True

        return False
