import gc
import logging
import os
import sys
import warnings
from typing import Optional

import absl.logging
import numpy as np
import seaborn as sns
import tensorflow as tf
import tf2onnx
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model, mixed_precision
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


# See: https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-768977280
# See also: https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
def get_model_flops(model: Model, write_path=None):
    '''
    Get total flops of current compiled model.

    Returns:
        (int): number of FLOPS
    '''
    # tf warnings have been disabled in manager.py, a lot of them are displayed in this function, but they seem irrelevant.
    # enable warnings in manager.py if you want to investigate for potential issues.
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        if write_path is not None:
            opts['output'] = 'file:outfile={}'.format(write_path)  # redirect output
        else:
            opts['output'] = 'none'

        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

    return flops.total_float_ops


def initialize_train_strategy(strategy_device: Optional[str], use_mixed_precision: bool) -> tf.distribute.Strategy:
    # debug available devices
    device_list = tf.config.list_physical_devices()
    print(device_list)

    gpu_devices = tf.config.list_physical_devices('GPU')
    tpu_devices = tf.config.list_physical_devices('TPU')

    missing_device_msg = f'{strategy_device} is not available for execution, run with a different train strategy' \
                         f' or troubleshot the issue in case a {strategy_device} is actually present in the device.'

    if use_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    # Generate the train strategy. Currently supported values: ['CPU', 'GPU', 'multi-GPU', 'TPU']
    # Basically a switch, which it's not supported in used python version.
    if strategy_device is None:
        # should use GPU if available, otherwise CPU. Fallback for when strategy is not specified and for backwards compatibility.
        train_strategy = tf.distribute.get_strategy()
    elif strategy_device == 'CPU':
        # remove GPUs from visible devices, using only CPUs
        tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices([], 'TPU')

        print('Using CPU devices only')
        # default strategy
        train_strategy = tf.distribute.get_strategy()
    elif strategy_device == 'GPU':
        if len(gpu_devices) == 0:
            sys.exit(missing_device_msg)

        # default strategy also for single GPU
        train_strategy = tf.distribute.get_strategy()
    elif strategy_device == 'multi-GPU':
        if len(gpu_devices) < 2:
            sys.exit(f'At least 2 GPUs required for multi-GPU strategy, {len(gpu_devices)} GPUs have been found.')

        # by default, use all GPUs found in the worker. Multiple workers (cluster of devices) are not currently supported.
        # NOTE: Tune the batch size and learning rate to exploit more parallelism, if necessary.
        train_strategy = tf.distribute.MirroredStrategy()
    elif strategy_device == 'TPU':
        if len(tpu_devices) == 0:
            sys.exit(missing_device_msg)

        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        train_strategy = tf.distribute.TPUStrategy(cluster_resolver)
    else:
        sys.exit('Train strategy provided in configuration file is invalid')

    policy = mixed_precision.global_policy()
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    return train_strategy


def get_optimized_steps_per_execution(train_strategy: tf.distribute.Strategy):
    '''
    Get a good steps_per_execution parameter for Keras model.compile.
    If the returned value is > batches, it is automatically trimmed by Keras model.fit.

    It is extremely important for optimizing TPU performances. Note that it works also at inference time.
    '''
    # TODO: is it useful only for TPUs or can we add more optimization cases?
    return 32 if isinstance(train_strategy, tf.distribute.TPUStrategy) else 1


def save_keras_model_to_onnx(model: Model, save_path: str):
    tf2onnx.convert.from_keras(model, opset=13, output_path=save_path)


def support_weight_decay(optimizer_str: str):
    return any(optimizer_str.startswith(opt) for opt in ['adamW', 'SGDW', 'radam'])


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, n_classes: int, normalize: bool):
    # if normalize is set, normalize the rows (true values)
    cmat = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    fig_size = 8 + 0.2 * n_classes
    fig = plt.figure(figsize=(1 + fig_size, fig_size))
    if n_classes <= 20:
        fmt = '.3f' if normalize else 'd'
        sns.heatmap(cmat, annot=True, fmt=fmt, square=True)
    # avoid annotations in case there are too many classes
    else:
        sns.heatmap(cmat, square=True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=120)
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def predict_and_save_confusion_matrix(model: Model, ds: tf.data.Dataset, multi_output: bool, n_classes: int, save_path: str):
    # Y labels must be converted to integers and flatten (since they are batched)
    y_pred = model.predict(x=ds)
    # take last output, in case the model is multi-output
    if multi_output:
        y_pred = y_pred[-1]
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    # DS values are stored as one-hot. Convert them to integer labels.
    y_true = np.concatenate([np.argmax(y, axis=-1) for x, y in ds], axis=0)

    # save both normalized and not normalized versions of confusion matrix
    save_confusion_matrix(y_true, y_pred, save_path=save_path, n_classes=n_classes, normalize=False)
    save_confusion_matrix(y_true, y_pred, save_path=f'{save_path}_norm', n_classes=n_classes, normalize=True)


def perform_global_memory_clear():
    ''' Clean up memory by forcing the deletion of python unreachable objects and clearing the Keras global state. '''
    collected_count = gc.collect()
    # print(f'GC collected {collected_count} items')
    # print(f'GC has {len(gc.garbage)} items which are garbage but cannot be freed')
    tf.keras.backend.clear_session()


def remove_annoying_tensorflow_messages():
    '''
    Disable multiple warnings and errors, which seems to be bugged or unclear how to address them.

    When performing major changes or updating Tensorflow, shortcut this function with a *return*, to check eventual problematic warnings.
    '''
    # return

    # disable Tensorflow messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # disable strange useless warning in model saving, that is also present in TF tutorial...
    absl.logging.set_verbosity(absl.logging.ERROR)

    # disable Tensorflow info and warning messages (Warning are not on important things, they were investigated. Still, enable them
    # when performing changes to see if there are new potential warnings that can affect negatively the algorithm).
    tf.get_logger().setLevel(logging.ERROR)

    # disable tf2onnx conversion messages
    tf2onnx.logging.set_level(logging.WARN)

    # Disable warning triggered at every training. get_config is actually implemented for each custom layer of ops module.
    # The module responsible for this annoying warning: tensorflow\python\keras\utils\generic_utils.py, line 494
    warnings.filterwarnings(action='ignore',
                            message='Custom mask layers require a config and must override get_config. '
                                    'When loading, the custom mask layer must be passed to the custom_objects argument.')
