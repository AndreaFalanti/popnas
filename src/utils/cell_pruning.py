# These classes are used for representing models in a more structured way, so that it's possible to compare
# them more easily (trim specular cells and equivalent models in general).
from search_space_units import CellSpecification


# TODO: name could be misleading, they are not actual encodings used by algorithm.
class BlockEncoding:
    def __init__(self, in1: str, op1: str, in2: str, op2: str) -> None:
        self.L = in1 + op1
        self.R = in2 + op2

    def __eq__(self, o: object) -> bool:
        if isinstance(o, BlockEncoding):
            return (self.L == o.L and self.R == o.R) or \
                   (self.L == o.R and self.R == o.L)
        return False


class CellEncoding:
    def __init__(self, model_list: CellSpecification) -> None:
        self.blocks = []

        for in1, op1, in2, op2 in model_list:
            # use another block output as input if input >=0
            # substitute input index with BlockEncoding in that case, otherwise keep it as int
            in1 = f'({self.blocks[in1].L}{self.blocks[in1].R})' if in1 >= 0 else str(in1)
            in2 = f'({self.blocks[in2].L}{self.blocks[in2].R})' if in2 >= 0 else str(in2)

            self.blocks.append(BlockEncoding(in1, op1, in2, op2))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, CellEncoding) and len(self.blocks) == len(o.blocks):
            # each block must be equivalent to a not already eqv matched block for having a complete cell equivalence.
            valid_blocks = o.blocks.copy()  # shallow copy, references of blocks no more valid will be removed

            for block_enc in self.blocks:
                eqv_to_other_cell_block = False
                # index produced by enumerate is required to update the boolean mask
                for valid_block in valid_blocks:
                    if block_enc == valid_block:
                        # this block is no more viable for further equalities
                        valid_blocks.remove(valid_block)
                        eqv_to_other_cell_block = True
                        break

                # if any block is not equivalent to a viable one, the cells are different
                if not eqv_to_other_cell_block:
                    return False

            return True

        return False


# Pruning functions


def prune_equivalent_cell_models(models: 'list[CellSpecification]', k: int):
    '''
    Prune equivalent models from given models list until k models have been obtained, then return them.
    Useful for pruning eqv models during PNAS mode children selection.

    Args:
        models: list of cell specifications
        k: model count

    Returns:
        (tuple): prime models list and int counter of total pruned models
    '''
    prime_models = []
    prime_cell_repr = []
    # pruned_models = []  # DEBUG only
    pruned_count = 0

    for model in models:
        # build a better cell representation for equivalence purposes
        cell_repr = CellEncoding(model)

        # check possible equivalence with other models already generated
        if is_model_equivalent_to_another(cell_repr, prime_cell_repr):
            pruned_count += 1
            # pruned_models.append(model)     # DEBUG only
        else:
            prime_cell_repr.append(cell_repr)
            prime_models.append(model)

        # reached k models, return them without considering the rest
        if len(prime_models) == k:
            break

    return prime_models, pruned_count


def is_model_equivalent_to_another(model: CellEncoding, other_models: 'list[CellEncoding]'):
    '''
    Check if model is equivalent to any model generated previously.

    Args:
        model: model to check
        other_models: previous models

    Returns:
        (bool): model is eqv to another one or not
    '''
    for gen_model in other_models:
        if model == gen_model:
            return True

    return False
