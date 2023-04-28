import re
from functools import cached_property
from logging import Logger
from typing import NamedTuple, Any, Iterable

from utils.func_utils import list_flatten


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

    def __eq__(self, o: object) -> bool:
        if isinstance(o, CellSpecification):
            return self._flat_data == o._flat_data
        return False

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

    def pretty_logging(self, logger: Logger):
        ''' Print the cell specification space properly '''
        logger.info('Cell specification:')
        for i, block in enumerate(self._data):
            logger.info("\tBlock %d: %s", i + 1, block)

    @cached_property
    def inputs(self) -> list:
        return self._flat_data[::2]

    @cached_property
    def operators(self) -> list:
        return self._flat_data[1::2]

    @cached_property
    def used_lookbacks(self) -> 'set[int]':
        return set(filter(lambda el: el < 0, self.inputs))

    @cached_property
    def used_blocks(self) -> 'set[int]':
        ''' Indexes of blocks which are used as input of at least another block inside the cell specification. '''
        return set(filter(lambda el: el >= 0, self.inputs))

    @cached_property
    def unused_blocks(self) -> 'set[int]':
        ''' Indexes of blocks which must be concatenated in the final cell output, since not used internally by other blocks. '''
        return set(range(len(self))) - self.used_blocks

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

    def prune_fictitious_blocks(self):
        return CellSpecification(b for b in self._data if b != (None, None, None, None))


def parse_cell_strings(cell_structures: Iterable[str]) -> 'list[CellSpecification]':
    '''
    Function used to parse in an actual python structure the csv field storing the non-encoded cell structure, which is saved in form:
    "[(in1,op1,in2,op2);(...);...]"
    '''
    return [CellSpecification.from_str(cell_str) for cell_str in cell_structures]
