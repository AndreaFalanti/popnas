import operator
import re
import sys
from functools import reduce

import numpy as np
import seaborn as sns
import tensorflow as tf
import tf2onnx
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


class TrainingResults:
    ''' Utility class for passing the training results of a networks, together with other interesting network characteristics. '''
    def __init__(self, cell_spec: 'list[tuple]', accuracy: float, f1_score: float, training_time: float, inference_time: float,
                 params: int, flops: int) -> None:
        self.cell_spec = cell_spec
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.training_time = training_time
        self.inference_time = inference_time
        self.params = params
        self.flops = flops

        self.blocks = len(cell_spec)

    @staticmethod
    def get_csv_headers():
        return ['best val accuracy', 'val F1 score', 'training time(seconds)', 'inference time(seconds)', 'total params',
                'flops', '# blocks', 'cell structure']

    def to_csv_list(self):
        ''' Return a list with fields ordered for csv insertion '''
        return [self.accuracy, self.f1_score, self.training_time, self.inference_time, self.params, self.flops, self.blocks, self.cell_spec]

    @staticmethod
    def from_csv_row(row: list) -> 'TrainingResults':
        ''' Some code parts use an ordered tuple (time, acc, cell_spec), so return it to support more easily their logic without refactors. '''
        return TrainingResults(row[7], *row[0:6])


def get_best_metric_per_output(hist: History, metric: str, maximize: bool = True):
    '''
    Produce a dictionary with a key for each model output.
     The value associated is the best one found for that output during the whole train procedure.
    Args:
        hist: train history
        metric: metric name
        maximize: True if max is best, false instead if min is best

    Returns:
        (dict[str, float]): dictionary with best metric values, for each output.
    '''
    r = re.compile(rf'val_Softmax_c(\d+)_{metric}')
    output_indexes = [int(match.group(1)) for match in map(r.match, hist.history.keys()) if match]

    # save best score reached for each output
    multi_output_scores = {}
    for output_index in output_indexes:
        multi_output_scores[f'c{output_index}_{metric}'] = max(hist.history[f'val_Softmax_c{output_index}_{metric}']) if maximize \
            else min(hist.history[f'val_Softmax_c{output_index}_{metric}'])

    return multi_output_scores


def get_multi_output_best_epoch_stats(hist: History, score_metric: str, using_val: bool = True):
    '''
    Produce a dictionary with a key for each model output, that contains the metrics of the best epoch for that output.
    Args:
        hist: train history
        score_metric:

    Returns:
        (int, float, dict[int, dict[str, float]]): best epoch index, best validation score and dictionary with epoch metrics for each output.
    '''
    metric_prefix = 'val_Softmax_c' if using_val else 'Softmax_c'
    r = re.compile(rf'{metric_prefix}(\d+)_{score_metric}')
    output_indexes = [int(match.group(1)) for match in map(r.match, hist.history.keys()) if match]

    # get the best validation score metric for each cell
    best_scores = [max(enumerate(hist.history[f'{metric_prefix}{output_index}_{score_metric}']), key=operator.itemgetter(1))
                   for output_index in output_indexes]
    # find epoch where max validation score is achieved
    best_epoch, best_val_acc = max(best_scores, key=operator.itemgetter(1))

    epoch_metrics_per_output = {}
    for output_index in output_indexes:
        epoch_metrics_per_output[output_index] = {}
        if using_val:
            epoch_metrics_per_output[output_index]['val_loss'] = hist.history[f'val_Softmax_c{output_index}_loss'][best_epoch]
            epoch_metrics_per_output[output_index]['val_acc'] = hist.history[f'val_Softmax_c{output_index}_accuracy'][best_epoch]
            epoch_metrics_per_output[output_index]['val_top_k'] = hist.history[f'val_Softmax_c{output_index}_top_k_categorical_accuracy'][best_epoch]
            epoch_metrics_per_output[output_index]['val_f1'] = hist.history[f'val_Softmax_c{output_index}_f1_score'][best_epoch]

        epoch_metrics_per_output[output_index]['loss'] = hist.history[f'Softmax_c{output_index}_loss'][best_epoch]
        epoch_metrics_per_output[output_index]['acc'] = hist.history[f'Softmax_c{output_index}_accuracy'][best_epoch]
        epoch_metrics_per_output[output_index]['top_k'] = hist.history[f'Softmax_c{output_index}_top_k_categorical_accuracy'][best_epoch]
        epoch_metrics_per_output[output_index]['f1'] = hist.history[f'Softmax_c{output_index}_f1_score'][best_epoch]

    return best_epoch, best_val_acc, epoch_metrics_per_output


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


def compute_tensor_byte_size(tensor: tf.Tensor):
    dtype_sizes = {
        tf.float32: 4,
        tf.float64: 8,
        tf.int32: 4,
        tf.int64: 8
    }

    dtype_size = dtype_sizes[tensor.dtype]
    # remove batch size from shape
    tensor_shape = tensor.get_shape().as_list()[1:]

    # byte size is: (number of weights) * (size of each weight)
    return reduce(operator.mul, tensor_shape, 1) * dtype_size


def initialize_train_strategy(config_strategy_device: str) -> tf.distribute.Strategy:
    # debug available devices
    device_list = tf.config.list_physical_devices()
    print(device_list)

    gpu_devices = tf.config.list_physical_devices('GPU')
    tpu_devices = tf.config.list_physical_devices('TPU')

    missing_device_msg = f'{config_strategy_device} is not available for execution, run with a different train strategy' \
                         f' or troubleshot the issue in case a {config_strategy_device} is actually present in the device.'

    # if tf.test.gpu_device_name():
    #     print('GPU found')
    # else:
    #     print('No GPU found')

    # TODO: add multi-GPU
    # Generate the train strategy. Currently supported values: ['CPU', 'GPU', 'TPU']
    # Basically a switch, but is not supported in python.
    if config_strategy_device is None:
        # should use GPU if available, otherwise CPU. Fallback for when strategy is not specified and for backwards compatibility.
        train_strategy = tf.distribute.get_strategy()
    elif config_strategy_device == 'CPU':
        # remove GPUs from visible devices, using only CPUs
        tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices([], 'TPU')
        print('Using CPU devices only')
        # default strategy
        train_strategy = tf.distribute.get_strategy()
    elif config_strategy_device == 'GPU':
        if len(gpu_devices) == 0:
            sys.exit(missing_device_msg)
        # default strategy also for single GPU
        train_strategy = tf.distribute.get_strategy()
    elif config_strategy_device == 'TPU':
        if len(tpu_devices) == 0:
            sys.exit(missing_device_msg)
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        train_strategy = tf.distribute.TPUStrategy(cluster_resolver)
    else:
        sys.exit('Train strategy provided in configuration file is invalid')

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
    tf2onnx.convert.from_keras(model, opset=10, output_path=save_path)


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
