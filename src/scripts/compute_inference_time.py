import argparse
import os.path
from ast import literal_eval
from statistics import mean
from timeit import default_timer as timer
from typing import Callable

import numpy as np
import onnxruntime
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image


def generate_synthetic_samples(input_size: 'tuple[int, ...]', num_samples: int):
    data = np.random.rand(num_samples, *input_size)
    return data.astype(np.float32)


def load_segmentation_samples(samples_path: str, samples_limit: int):
    if samples_path.endswith('.npz'):
        return load_segmentation_array(samples_path, samples_limit)
    elif os.path.isdir(samples_path):
        return load_segmentation_images(samples_path, samples_limit)
    else:
        raise ValueError('Dataset path must be either a folder with images or the path to a .npz file')


def load_segmentation_images(samples_path: str, samples_limit: int):
    ''' Load images from file, convert them into numpy arrays, and apply the preprocessing to convert RGB int values to float in [0, 1]. '''
    sample_files = [f for f in os.scandir(samples_path) if f.is_file()]  # type: list[os.DirEntry]

    samples = []
    for i in range(samples_limit):
        with Image.open(sample_files[i].path) as img:
            x = np.asarray(img) / 255.
            samples.append(x)

    return np.asarray(samples, dtype=np.float32)


def load_segmentation_array(npz_path: str, samples_limit: int):
    ''' Load images from numpy npz file, and apply the preprocessing to convert RGB int values to float in [0, 1]. '''
    with np.load(npz_path) as npz:
        samples = npz['x'][:samples_limit]  # type: np.ndarray
        samples = samples / 255.

    return samples.astype(np.float32)


def format_list(elems: list):
    return ', '.join(f'{t:.4f}' for t in elems)


def print_results(inference_times: list, model_name: str):
    print(f'{model_name} inference times: {format_list(inference_times)}s')
    mean_inference_time = mean(inference_times)
    print(f'Mean {model_name} inference time per batch: {mean_inference_time:.4f}s')
    print(f'{model_name} FPS: {1 / mean_inference_time:.1f}')


def perform_model_inference(test_batch_size: int, num_batches: int, x: np.ndarray, predict_f: Callable[[np.ndarray, int], np.ndarray]):
    ''' Perform the inference on the provided test batches, logging the time required for each batch. '''
    inf_times = []
    for i in range(num_batches):
        test_data = x[test_batch_size * i:test_batch_size * (i + 1)]

        start_time = timer()
        # use [1] because the models have two outputs (intermediate output at 2/3 of the network and final cell output), take the last one
        preds = predict_f(test_data, test_batch_size)[1]
        inf_times.append(timer() - start_time)

    return inf_times


def perform_io_binding_onnx_inference(onnx_session: onnxruntime.InferenceSession, test_batch_size: int, num_batches: int, x: np.ndarray):
    '''
    Perform the inference on the provided test batches, logging the time required for each batch.

    Optimized for ONNX runtime, it places input and output buffers on GPU before the execution, making it possible to observe just the
    computation time required by the network.
    '''
    inf_times = []
    for i in range(num_batches):
        test_data = x[test_batch_size * i:test_batch_size * (i + 1)]

        io_binding = prepare_onnx_io_binding(onnx_session, test_data)

        start_time = timer()
        onnx_session.run_with_iobinding(io_binding)
        inf_times.append(timer() - start_time)

    return inf_times


def prepare_onnx_io_binding(onnx_session: onnxruntime.InferenceSession, data: np.ndarray):
    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(data, 'cuda', 0)
    # output shape can either be the same of input but with different channels (one-hot of classes) or just one-hot of classes,
    # depending on the type of task addressed (i.e., classification or segmentation)
    # automatically infer the task and set the right shape from the number of dimensions (discard first, which is batch)
    out_dims = onnx_session.get_outputs()[-1].shape[1:]
    num_classes = out_dims[-1]
    # classification, (batch_size, one-hot of classes)
    if len(out_dims) == 1:
        output_shape = [data.shape[0], num_classes]
    # segmentation, use input shape for setting the resolution and batch size
    else:
        output_shape = list(data.shape)[:-1] + [num_classes]

    y1_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(output_shape, np.float32, 'cuda', 0)
    y2_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(output_shape, np.float32, 'cuda', 0)

    io_binding = onnx_session.io_binding()
    # extract the name of the input layer (always single input in POPNAS)
    in_name = onnx_session.get_inputs()[0].name
    # output names can also change based on model structure (but are always two outputs for post-search models)
    out_1_name, out_2_name = [out.name for out in onnx_session.get_outputs()]

    io_binding.bind_input(name=in_name, device_type=x_ortvalue.device_name(), device_id=0, element_type=data.dtype, shape=x_ortvalue.shape(),
                          buffer_ptr=x_ortvalue.data_ptr())
    io_binding.bind_output(name=out_1_name, device_type=y1_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=y1_ortvalue.shape(),
                           buffer_ptr=y1_ortvalue.data_ptr())
    io_binding.bind_output(name=out_2_name, device_type=y2_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=y2_ortvalue.shape(),
                           buffer_ptr=y2_ortvalue.data_ptr())

    return io_binding


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to model folder", required=True)
    parser.add_argument('-d', metavar='DATASET_PATH', type=str, help="path to dataset folder split")
    parser.add_argument('-i', metavar='INPUT_SIZE', type=str, help="input size as a tuple string (no batch dim), e.g., '(400, 400, 3)'")
    parser.add_argument('-b', metavar='BATCH_SIZE', type=int, help="batch size used for inference", default=1)
    parser.add_argument('-n', metavar='NUM_BATCHES', type=int, help="number of batches to process", default=100)
    parser.add_argument('--fp16', help='specify TensorRT to enable fp16 execution', action='store_true')
    parser.add_argument('--profile', help='profile ONNX execution', action='store_true')
    parser.add_argument('--onnx_only', help='execute tests only on the ONNX model', action='store_true')
    args = parser.parse_args()

    test_batch_size = args.b
    # load 10 extra batches, which will be used to warm up the model and the GPU
    warmup_batches = 10
    num_batches = args.n + warmup_batches
    num_samples = test_batch_size * num_batches

    if args.d is not None:
        samples = load_segmentation_samples(samples_path=args.d, samples_limit=num_samples)
    elif args.i is not None:
        input_size = literal_eval(args.i)
        samples = generate_synthetic_samples(input_size, num_samples)
    else:
        raise ValueError('Either the dataset or the input size argument must be provided to perform the inference test')

    # ONNX model: make sure to install onnxruntime-gpu, otherwise remove CUDA provider
    # produce a json with profiling data, if the related flag is set
    if args.profile:
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
    else:
        sess_options = None

    onnx_providers = [
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': args.fp16
        }),
        ('CUDAExecutionProvider', {
            'cudnn_conv_use_max_workspace': '1'
        })
    ]
    # onnx_providers = ['CUDAExecutionProvider']

    sess = onnxruntime.InferenceSession(os.path.join(args.p, 'trained.onnx'), providers=onnx_providers, sess_options=sess_options)
    # extract the name of the input layer (always single input in POPNAS)
    sess_input_layer = sess.get_inputs()[0].name
    onnx_times = perform_model_inference(test_batch_size, num_batches, samples,
                                         predict_f=lambda data, size: sess.run(None, {sess_input_layer: data}))
    print_results(onnx_times[warmup_batches:], 'ONNX')

    onnx_times = perform_io_binding_onnx_inference(sess, test_batch_size, num_batches, samples)
    print_results(onnx_times[warmup_batches:], 'ONNX IO bound')

    # TF model
    if not args.onnx_only:
        custom_objects = {
            'Addons>Lookahead': tfa.optimizers.Lookahead
        }
        tf_model = tf.keras.models.load_model(os.path.join(args.p, 'tf_model'), custom_objects)  # type: tf.keras.Model
        tf_times = perform_model_inference(test_batch_size, num_batches, samples,
                                           predict_f=lambda data, size: tf_model.predict(data, batch_size=size))
        print_results(tf_times[warmup_batches:], 'TF')


if __name__ == '__main__':
    main()
