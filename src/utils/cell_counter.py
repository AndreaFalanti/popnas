from collections import Counter

from utils.func_utils import list_flatten


def _get_counter_total(counter: Counter):
    return sum(counter.values())


def _remove_unwanted_keys(counter: Counter, key_list: list):
    # deep copy keys to avoid changes during iteration (that would lead to RuntimeError)
    keys = [key for key in counter.keys()]
    for key in keys:
        if key not in key_list:
            del counter[key]


def _initialize_counter_values(counter: Counter, key_list: list):
    for key in key_list:
        counter[key] = 0


class CellCounter:
    def __init__(self, input_keys: list = None, op_keys: list = None) -> None:
        '''
        Wrapper for python Counters, designed specifically for operating on cell specifications.
        input_keys and op_keys allow to count only some specific values, making the total and key_count functions return
        info only about those keys.

        Internal counters are accessible for more flexibility, but it is not advised to update them directly, use instead
        the update_from_cell_spec method.

        Args:
            input_keys: list of input to count and store in counter dict. If None, all inputs are counted.
            op_keys: list of operators to count and store in counter dict. If None, all operators are counted.
        '''
        self.input_keys = input_keys
        self.op_keys = op_keys

        self.input_counter = Counter()
        self.op_counter = Counter()

        # initialize values to 0, otherwise the unseen elements will not be inserted in counter dictionary,
        # resulting in problems regarding the key_count and items (0 values should be returned, if keys are explicit).
        if input_keys is not None:
            _initialize_counter_values(self.input_counter, self.input_keys)
        if op_keys is not None:
            _initialize_counter_values(self.op_counter, self.op_keys)

    def update_from_cell_spec(self, cell_spec: 'list[tuple]'):
        flat_cell = list_flatten(cell_spec)
        cell_inputs = flat_cell[::2]    # even indexes
        cell_ops = flat_cell[1::2]      # odd indexes

        # avoid updating if not interested in counters for either inputs or operators (more efficient)
        if self.input_keys is None or len(self.input_keys) > 0:
            self.input_counter.update(cell_inputs)
        if self.op_keys is None or len(self.op_keys) > 0:
            self.op_counter.update(cell_ops)

        # removing undesired keys guarantees that total and key_count are the values truly expected
        if self.input_keys is not None:
            _remove_unwanted_keys(self.input_counter, self.input_keys)
        if self.op_keys is not None:
            _remove_unwanted_keys(self.op_counter, self.op_keys)

    def inputs_total(self):
        return _get_counter_total(self.input_counter)

    def ops_total(self):
        return _get_counter_total(self.op_counter)

    def inputs_keys_len(self):
        return len(self.input_counter.keys())

    def ops_keys_len(self):
        return len(self.op_counter.keys())
