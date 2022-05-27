from typing import Counter

from encoder import SearchSpace
from utils.cell_counter import CellCounter
from utils.func_utils import get_valid_inputs_for_block_size
from utils.model_estimate import ModelEstimate


def compute_exploration_value_sets(pareto_front_models: 'list[ModelEstimate]', search_space: SearchSpace, current_b: int, B: int):
    valid_inputs = get_valid_inputs_for_block_size(search_space.input_values, current_b, B)
    valid_ops = search_space.operator_values
    cell_counter = CellCounter(valid_inputs, valid_ops)

    for model in pareto_front_models:
        cell_counter.update_from_cell_spec(model.cell_spec)

    op_usage_threshold = cell_counter.ops_total() / (len(valid_ops) * 5)
    input_usage_threshold = cell_counter.inputs_total() / (len(valid_inputs) * 5)

    op_exp = set(key for key, val in cell_counter.op_counter.items() if val < op_usage_threshold)
    input_exp = set(key for key, val in cell_counter.input_counter.items() if val < input_usage_threshold)

    return op_exp, input_exp


def get_block_element_exploration_score(el, exploration_counter: Counter, total_count: int, bonus: bool):
    score = 0

    # el in exploration set (dict is initialized with valid keys)
    if el in exploration_counter.keys():
        score += 1

        # el underused condition (less than average). If only one element is present, it will be always True.
        if total_count == 0 or exploration_counter[el] <= (total_count / len(exploration_counter.keys())):
            score += 2

        if bonus:
            score += 1

    return score


def has_sufficient_exploration_score(model_est: ModelEstimate, exp_cell_counter: CellCounter,
                                     exploration_pareto_front: 'list[ModelEstimate]'):
    exp_score = 0
    exp_inputs_total_count = exp_cell_counter.inputs_total()
    exp_ops_total_count = exp_cell_counter.ops_total()

    # give a bonus to the least searched set between inputs and operators (to both if equal)
    # in case one exploration set is empty, after first step the bonus will not be granted anymore.
    op_bonus = exp_ops_total_count <= exp_inputs_total_count
    input_bonus = exp_inputs_total_count <= exp_ops_total_count

    for in1, op1, in2, op2 in model_est.cell_spec:
        exp_score += get_block_element_exploration_score(in1, exp_cell_counter.input_counter, exp_inputs_total_count, input_bonus)
        exp_score += get_block_element_exploration_score(in2, exp_cell_counter.input_counter, exp_inputs_total_count, input_bonus)
        exp_score += get_block_element_exploration_score(op1, exp_cell_counter.op_counter, exp_ops_total_count, op_bonus)
        exp_score += get_block_element_exploration_score(op1, exp_cell_counter.op_counter, exp_ops_total_count, op_bonus)

    # additional conditions for pareto variety (float values, 1 point every difference of 4% of accuracy or 10% of time difference
    # with previous pareto entry). Considered only for cells with elements in exploration sets, when exploration pareto front is not empty.
    if exp_score > 0 and len(exploration_pareto_front) > 0:
        exp_score += (1 - model_est.score / exploration_pareto_front[-1].score) / 0.04
        exp_score += (1 - model_est.time / exploration_pareto_front[-1].time) / 0.10

    # adapt threshold if one of the two sets is empty
    exp_score_threshold = 8 if (exp_cell_counter.ops_keys_len() > 0 and exp_cell_counter.inputs_keys_len() > 0) else 4
    return exp_score >= exp_score_threshold
