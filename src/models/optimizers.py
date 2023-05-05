import re
from typing import Union, TypeVar

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow_addons import optimizers as tfa_optimizers

from models.custom_schedulers import WarmUpSchedulerWrapper
from utils.config_dataclasses import OptimizerDict

decimal_regex = r'\d+.?\d*'

T = TypeVar('T', int, str, float)
Rate = Union[float, schedules.LearningRateSchedule]


def get_group_or_default(match: re.Match, group_id: Union[int, str], default: T) -> T:
    group_val = match.group(group_id)
    # tricky code to convert "group_val" to the same type of the default value provided
    return default if group_val is None else type(default)(group_val)


def build_scheduler(scheduler_str: str, learning_rate: float, weight_decay: float, training_steps_per_epoch: int, epochs: int) -> 'tuple[Rate, Rate]':
    # fixed rates (no schedules)
    if scheduler_str is None or scheduler_str == '':
        return learning_rate, weight_decay

    # Cosine Decay Restart
    match = re.compile(
        rf'cdr:?(?: (?P<period>\d+) period)?,?(?: (?P<t_mul>{decimal_regex}) t_mul)?,?(?: (?P<m_mul>{decimal_regex}) m_mul)?,?(?: (?P<alpha>{decimal_regex}) alpha)?') \
        .match(scheduler_str)
    if match:
        period = get_group_or_default(match, 'period', 3)
        t_mul = get_group_or_default(match, 't_mul', 2.0)
        m_mul = get_group_or_default(match, 'm_mul', 1.0)
        alpha = get_group_or_default(match, 'alpha', 0.0)
        decay_steps = training_steps_per_epoch * period

        return schedules.CosineDecayRestarts(learning_rate, decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha), \
            schedules.CosineDecayRestarts(weight_decay, decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha)

    # Cosine Decay
    match = re.compile(rf'cd:?(?: (?P<alpha>{decimal_regex}) alpha)?').match(scheduler_str)
    if match:
        total_training_steps = training_steps_per_epoch * epochs
        alpha = get_group_or_default(match, 'alpha', 0.0)

        return schedules.CosineDecay(learning_rate, total_training_steps, alpha=alpha), \
            schedules.CosineDecay(weight_decay, total_training_steps, alpha=alpha)

    raise AttributeError(f'Scheduler "{scheduler_str}" not recognized')


def apply_warmup_to_schedulers(lr: Rate, wd: Rate, warmup_steps: int, max_learning_rate: float, max_weight_decay: float):
    return WarmUpSchedulerWrapper(lr, warmup_steps, max_learning_rate), WarmUpSchedulerWrapper(wd, warmup_steps, max_weight_decay)


def build_optimizer(optimizer_str: str, learning_rate: Rate, weight_decay: Rate, total_training_steps: int) -> optimizers.Optimizer:
    # AdamW
    match = re.compile(r'adamW').match(optimizer_str)
    if match:
        return tfa_optimizers.AdamW(weight_decay, learning_rate)

    # Adam
    match = re.compile(r'adam').match(optimizer_str)
    if match:
        return optimizers.Adam(learning_rate)

    # SGDW
    match = re.compile(rf'SGDW:?(?: (?P<momentum>{decimal_regex}) momentum)?').match(optimizer_str)
    if match:
        momentum = get_group_or_default(match, 'momentum', 0.0)
        return tfa_optimizers.SGDW(weight_decay, learning_rate, momentum)

    # SGD
    match = re.compile(rf'SGD:?(?: (?P<momentum>{decimal_regex}) momentum)?').match(optimizer_str)
    if match:
        momentum = get_group_or_default(match, 'momentum', 0.0)
        return optimizers.SGD(learning_rate, momentum)

    # Radam
    match = re.compile(rf'radam:?(?: (?P<alpha>{decimal_regex}) alpha)?').match(optimizer_str)
    if match:
        alpha = get_group_or_default(match, 'alpha', 0.0)
        return tfa_optimizers.RectifiedAdam(learning_rate, weight_decay=weight_decay, total_steps=total_training_steps,
                                            min_lr=learning_rate * alpha, warmup_proportion=0.1)


def instantiate_optimizer_and_schedulers(optimizer_config: OptimizerDict, learning_rate: float, weight_decay: float,
                                         training_steps_per_epoch: int, epochs: int) -> optimizers.Optimizer:
    warmup_epochs = optimizer_config.warmup

    lr, wd = build_scheduler(optimizer_config.scheduler, learning_rate, weight_decay, training_steps_per_epoch, epochs - warmup_epochs)
    if warmup_epochs > 0:
        lr, wd = apply_warmup_to_schedulers(lr, wd, warmup_epochs * training_steps_per_epoch, learning_rate, weight_decay)

    optimizer = build_optimizer(optimizer_config.type, lr, wd, training_steps_per_epoch * epochs)

    lookahead = optimizer_config.lookahead
    if lookahead is None:
        return optimizer
    else:
        return tfa_optimizers.Lookahead(optimizer, lookahead.sync_period, lookahead.slow_step_size)
