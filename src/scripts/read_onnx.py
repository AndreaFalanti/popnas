import argparse

import numpy as np
import onnxruntime
from tensorflow.keras import datasets


# TODO: this is just an example on how to load the final ONNX model and perform some inference, but it has no actual utility right now.
#  Also, it is hardcoded for CIFAR10.
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to ONNX model", required=True)
    args = parser.parse_args()

    sess = onnxruntime.InferenceSession(args.p)
    (x_train, y_train), _ = datasets.cifar10.load_data()
    x_train = x_train / 255.    # type: np.ndarray
    x_train = x_train.astype(dtype=np.float32)

    # use [1] because the models have two outputs, take the last one as indication (last cell output)
    pred_onnx = sess.run(None, {'input_1': x_train[0:5]})[1]
    for i, pred in enumerate(pred_onnx):
        print(f'Image {i}')
        print(f'\tSoftmax: {pred}')
        print(f'\tPredicted label: {np.argmax(pred, axis=-1)}')
        print(f'\tActual label: {y_train[i]}')


if __name__ == '__main__':
    main()
