from typing import Union

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WarmUpSchedulerWrapper(LearningRateSchedule):
    def __init__(self, main_scheduler: Union[LearningRateSchedule, float], warmup_steps: int,
                 target_lr: float, start_lr: float = 0.0, hold_steps: int = 0):
        '''
        Wraps a Keras scheduler instance and applies an initial warmup for the given number of steps,
        where the learning rate increases linearly to the target value.

        The warmup can stabilize the training of the network, especially when high learning rates are used.

        Args:
            main_scheduler: Keras scheduler to wrap, executed after the warmup is complete
            warmup_steps: initial steps used for warmup (number of weight updates)
            target_lr: LR reached after the warmup
            start_lr: LR used in the first step (default: 0)
            hold_steps: if > 0, the LR will be kept at the target LR after warmup for the specified number of steps
        '''
        super().__init__()

        # if a fixed learning rate (float) is provided, convert it into a callable
        self.main_scheduler = main_scheduler if isinstance(main_scheduler, LearningRateSchedule) else lambda step: main_scheduler
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps

    def __call__(self, step):
        # the main scheduler starts after warmup, also avoid potential problems from negative steps
        main_scheduler_steps = tf.maximum(step - self.warmup_steps, 0)
        main_scheduler_lr = self.main_scheduler(main_scheduler_steps)

        # warmup is linear between start_lr and target_lr, can also be hold for additional steps
        warmup_lr = self.start_lr + (self.target_lr - self.start_lr) * tf.minimum(step / self.warmup_steps, 1.0)
        warmup_lr = tf.cast(warmup_lr, tf.float32)

        # if step is in range of warmup + hold, then use warmup_lr, otherwise use the main scheduler one
        return tf.where(step < self.warmup_steps + self.hold_steps, warmup_lr, main_scheduler_lr, name='learning_rate')

    def get_config(self):
        config = super().get_config()
        config.update({
            'start_lr': self.start_lr,
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'hold_steps': self.hold_steps
        })
        return config
