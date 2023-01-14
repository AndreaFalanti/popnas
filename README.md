# POPNASv3
Third version of the Pareto-Optimal Progressive Neural Architecture Search (POPNAS) algorithm, a neural architecture search method based on 
[PNAS](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf), and direct extension
and improvement of the second version.

POPNASv2 has been developed by Andrea Falanti for his master's thesis at Politecnico di Milano, and the work has also been
published at the IEEE IJCNN 2022.
The paper and cite info are available at: https://ieeexplore.ieee.org/abstract/document/9892073.
The second version improves the time efficiency of the search algorithm, passing from an average 2x speed-up to
an average 4x speed-up compared to PNAS on the same experiment configurations.
The top-accuracy neural network architectures found are competitive with PNAS and other state-of-the-art methods,
solving the main drawback of the first POPNAS version.

POPNASv3 extends the second version by addressing time series classification problems, adding new operators like LSTM, GRU
and dilated convolutions to find network suited for these datasets.
The general macro-architecture is improved by adding configuration options for residual connections and direct reshaping of the lookbacks
in the block operators.
Furthermore, the training procedure now supports multiple hardware environments, such as single GPU, multiple GPUs with a synchronized replica
strategy and even TPU devices.
Another major contribution is the addition of post-search procedures, namely the _model selection_ and _final training_ steps.
The model selection trains more extensively the top networks found during the search, tuning also their macro-architecture.
The final training finalizes the process by training to convergence the best configuration found during the model selection,
producing a deployment-ready neural network saved in ONNX format.
You can read more about the latest version and its experiment results on the preprint available at: https://doi.org/10.48550/arXiv.2212.06735.

## Installation
This section provides information for installing all needed software and packages for properly run POPNASv3 on your system. If you prefer, you can
use the provided Docker container and skip these steps (see the Docker section below).

### Required additional software and tools
Virtual environment and dependencies are managed by _poetry_, check out its [repository](https://github.com/python-poetry/poetry)
for installing it on your machine and learning more about it.
Version >= 1.2.0 is required.

You need either python version 3.7 or 3.8 installed in your system for building a valid environment (python versions > 3.8 could work,
but they have not been officially tested).
To install and manage the python versions and work with _poetry_, it is advised to use [pyenv](https://github.com/pyenv/pyenv)
or [pyenv-win](https://github.com/pyenv-win/pyenv-win) based on your system.

Make also sure that *graphviz* is installed in your machine, since it is required to generate the plots of Keras models.
Follow the installation instructions at: https://graphviz.gitlab.io/download/.

### Installation steps
After installing poetry, optionally run on your terminal:
```
poetry config virtualenvs.in-project true
```
to change the virtual environment generation position directly in the project folder if you prefer so.

To generate the environment, open your terminal inside the repository folder and make sure that the python in use is a compatible version.
If not, you can install it and activate it through pyenv with these commands:
```
pyenv install 3.7.4 # or any valid version
pyenv shell 3.7.4
```

To install the dependencies, simply run:
```
poetry install
```
Poetry will generate a virtual env based on the active python version and install all the packages there.

You can activate the new environment with command:
```
poetry shell
```

The _aMLLibrary_ dependency is integrated in the project as a git submodule, since it is not possible to install it with package managers.
_aMLLibrary_ dependencies are handled by Poetry, but the library must be downloaded separately with the command:
```
git submodule update --init --recursive
```

After that you should be able to run POPNASv3 with this command:
```
python run.py
```

### GPU support
To enable GPU computations locally, you must satisfy Tensorflow GPU hardware and software requirements.
Follow https://www.tensorflow.org/install/gpu instructions to set up your device. Make sure
to install the exact versions of CUDA and CUDNN for Tensorflow 2.7 (see https://www.tensorflow.org/install/source#linux).

## Build Docker container
In the _docker_ folder, it is provided a dockerfile to extend an official Tensorflow container with pip packages required by the project
and finally mounting POPNAS source code.

To build the image, open the terminal into the root folder and execute this command:
```
docker build -f docker/Dockerfile -t andreafalanti/popnas:tf2.7.3 .
```

POPNASv3 can then be launched with command (set arguments as you like):
```
docker run -it --name popnas andreafalanti/popnas:tf2.7.3 python run.py -j configs/run_debug.json
```

## Run configuration
### Command line arguments
All command line arguments are optional.
- **-j**: specifies the path of the json configuration to use. If non provided, _configs/run.json_ will be used.
- **-r**: used to restore a previous interrupted run. Specifies the path of the log folder of the run to resume.
- **--name**: specifies a custom name for the log folder. If not provided, it defaults to date-time
  in which the run is started.

### Json configuration file
The run behaviour can be customized through the usage of custom json files. By default, the _run.json_ file
inside the _configs_ folder will be used. This file can be used as a template and customized to generate new configurations.
A properly structured json config file can be used by the algorithm by specifying its path in -j command line arguments.

Here it is presented a list of the configuration sections and fields, with a brief description.

**Search Space**:
- **blocks**: defines the maximum number of blocks a cell can contain.
- **lookback_depth**: maximum lookback depth to use (in absolute value). Lookback inputs are associated to previous cells,
  where _-1_ refers to last generated cell, _-2_ a skip connection to second-to-last cell, etc...
- **operators**: list of operators that can be used inside each cell. Note that the string format is important,
  since they are recognized by regexes.
  The currently supported operators, with customizable integer parameters(@) for kernel size and other parameters based on the operation type, are:
  - identity
  - @x@:@dr dconv (Depthwise-separable convolution)
  - @x@-@x@:@dr conv (Stacked convolutions)
  - @x@ conv
  - @x@ maxpool
  - @x@ avgpool
  - @x@ tconv (Transpose convolution)
  - (2D only) @k-@h-@b cvt (Convolutional Vision Transformer)
  - (2D only) @k-@h scvt (Simplified Convolutional Vision Transformer, custom operator not from literature)
  - (1D only) lstm
  - (1D only) gru

  conv and dconv support an optional group _:@dr_ for setting the dilation rate, which can be omitted  to use non-dilated convolutions.

  For time series (1D inputs), specify the kernel size as @ instead of @x@, since the kernel size is mono dimensional.

**Search strategy**:
- **max_children**: defines the maximum number of cells the algorithm can train in each iteration
  (except the first step, which trains all possible cells).
- **max_exploration_children**: defines the maximum number of cells the algorithm can train in the exploration step.
- **score_metric**: specifies the metric used for estimating the prediction quality of the trained models.
  Currently supported: [accuracy, f1_score].
- **additional_pareto_objectives**: defines the additional objectives considered during the search alongside the score metric, for optimizing
  the selection of the neural network architectures to train. Currently supported values: [time, params].
  POPNAS requires at least one of them.
  
**CNN hyperparameters**:
- **epochs**: defines for how many epochs E each child network has to be trained.
- **learning_rate**: defines the learning rate of the child CNN networks.
- **filters**: defines the initial number of filters to use, which increase in each reduction cell.
- **weight_reg**: defines the L2 regularization factor to use in CNNs. If _null_, regularization is not applied.
- **use_adamW**: use adamW instead of standard L2 regularization.
- **drop_path_prob**: defines the max probability of dropping a path in _scheduled drop path_. If set to 0,
  then _scheduled drop path_ is not used.
- **cosine_decay_restart**: dictionary for hyperparameters about cosine decay restart schedule.
  - **enabled**: if _true_ use cosine decay restart schedule, _false_ instead of using a cosine decay schedule.
  - **period_in_epochs**: first decay period in epochs, changes at each period based on _m_mul_ value.
  - **[t_mul, m_mul, alpha]**:
    see [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts).
- **softmax_dropout**: probability of dropping a value in output Softmax layer. If set to 0, then dropout is not used.
  Note that dropout is used on each output when _multi_output_ flag is set.


**CNN architecture parameters**:
- **motifs**: number of motifs to stack in each CNN. In NAS literature, a motif usually refers to a single cell,
  here instead it is used to indicate a stack of N normal cells followed by a single reduction cell.
- **normal_cells_per_motif**: normal cells to stack in each motif.
- **block_join_operator**: defines the operator used to join the tensors produced by the two operators of a block.
  Supported values: [add, avg].
- **lookback_reshape**: if _true_, when a skip lookback (-2) has a different shape from the one expected by the current cell, it is reshaped
  with a pointwise convolution, before passing it to block operators (as done in PNAS). If _false_, the skip lookbacks are instead
  passed directly to block operators requesting them, which would operate as reduction cell case when the shape diverges from expected one.
- **concat_only_unused_blocks**: if _true_, only blocks' output not used internally by the cell
  will be used in final concatenation cell output, following PNAS and NASNet.
  If set to _false_, all blocks' output will be concatenated in final cell output.
- **residual_cells**: if _true_, each cell output will be followed by a sum with the nearest lookback input used, to make the output residual
  (see ResNet paper). If lookback and cell output shapes diverge, pointwise convolution and/or max pooling are performed to adapt 
  the spatial dimension and the filters.
- **multi_output**: if _true_, CNNs will have an output exit (GAP + Softmax) at the end of each cell.

[//]: # (**RNN hyperparameters &#40;controller, optional&#41;**: \\)

[//]: # (If the parameters are not provided or the object is omitted in JSON config, default parameters will be applied.)

[//]: # (The accepted parameters and their names depend on the model type chosen for the controller.)

[//]: # (See KerasPredictor subclasses to have a better idea &#40;TODO&#41;.)

[//]: # (- **epochs**: how many epochs the LSTM is trained on, at each expansion step.)

[//]: # (- **lr**: LSTM learning rate.)

[//]: # (- **wr**: LSTM L2 weight regularization factor. If _null_, regularization is not applied.)

[//]: # (- **er**: LSTM L2 weight regularization factor applied on embeddings only. If _null_, regularization is not applied.)

[//]: # (- **embedding_dim**: LSTM embedding dimension, used for both inputs and operator embeddings.)

[//]: # (- **cells**: total LSTM cells of the model.)

**Dataset**:
- **type**: specifies the problem domain of the provided data. Processing pipeline and other parameters depends on the
  addressed task. Supported values: [image_classification, time_series_classification].
- **name**: used to identify and load a Keras or TFDS dataset supported by POPNAS.
  Can be _null_ if the path of a custom dataset is provided.
- **path**: path to a folder containing a custom dataset.
  Can be _null_ if you want to use a supported dataset already present in Keras or TFDS.
- **classes_count**: classes present in the dataset. If using a Keras dataset, this value can be inferred automatically.
- **batch_size**: defines the batch size dimension of the dataset.
- **inference_batch_size**: defines the batch size dimension for benchmarking the inference time of a network.
- **validation_size**: fraction of the total samples to use for the validation set, e.g. _0.1_ value means that 10% of the
  training samples will be reserved for the validation dataset. Can be _null_ for TFDS dataset which have
  a separated validation set, for using it instead of partitioning the training set.
- **cache**: if _true_, the dataset will be cached in memory, increasing the training performance.
  Strongly advised for small datasets.
- **folds**: number of dataset folds to use. When using multiple folds, the metrics extrapolated from CNN training
  will be the average of the ones obtained on each fold.
- **samples**: if provided, limits the total dataset samples used by the algorithm to the number provided (integer).
  Useful for fast testing.
- **balance_class_losses**: if _true_, the class losses will be weighted proportionally to the number of samples.
  The exact formula for computing the weight of each class is:
  w<sub>class</sub> = 1 / (classes_count * samples_fraction<sub>class</sub>).
- **data_augmentation**: dictionary with parameters related to data augmentation.
  - **enabled**: _true_ for using data augmentation, _false_ otherwise.
  - **perform_on_gpu**: perform data augmentation directly on GPU (through Keras experimental layers).
    Usually advised only if CPU is very slow, since CPU prepares the images while the GPU trains the network
    (asynchronous prefetch), instead performing data augmentation on the GPU will make the process sequential,
    always causing delays even if it's faster to perform on GPU.
- ...extra parameters depending on dataset type, see next sections.

**Dataset(_image_classification_ only)**:
- **resize**: dictionary with parameters related to image resizing.
  - **enabled**: _true_ for using resizing, _false_ otherwise.
  - **width**: target image width in pixels.
  - **height**: target image height in pixels.

**Dataset(_time_series_classification_ only)**:
- **rescale**: if _true_ the values will be rescaled with a factor based on 98 percentile of the entire input values.
- **normalize**: if _true_, sample values will be shifted and scaled into a distribution centered around 0 with standard deviation 1 (z-normalization).

**Others**:
- **accuracy_predictor_ensemble_units**: defines the number of models used in the accuracy predictor (ensemble).
- **predictions_batch_size**: defines the batch size used when performing both time and accuracy predictions in the controller
  update step (predictions about cell expansions for blocks b+1). Incrementing it should decrease the prediction time
  linearly, up to a certain point, defined by hardware resources used.
- **save_children_weights**: if _true_, best weights of each child neural network are saved in log folder.
- **save_children_as_onnx**: if _true_, each child neural network will be serialized and saved as ONNX format.
- **pnas_mode**: if _true_, the algorithm will not use most of the improvements introduced by POPNAS, mainly the
  temporal regressor, Pareto optimality and exploration step, making the search process very similar to PNAS.
- **train_strategy**: defines the type of device and distribution strategy used for training the architectures sampled by the algorithm.
  Currently, it supports only local training with a single device. Accepted values: [CPU, GPU, multi-GPU, TPU].


## Output folder structure
Each run produces a single output folder, which contains all the files related to the run results.
Some files are generated only if the related configuration flag is set to true, refer to the JSON configuration file.

The files are organized in different subfolders:
- **best_model**: contains the checkpoint of the best model found during the search process.
- **csv**: contains many csv files with data extrapolated during the run, like the predictions and the training results.
- **plots**: contains all the plots automatically generated by the algorithm.
- **predictors**: contains logs and results about the predictors training process.
- **restore**: contains additional information for restoring an interrupted run. Contains also the input configuration file.
- **tensorboard_cnn**: contains tensorboard logs, model structure and summary of each neural network trained by the algorithm.


## Additional scripts and utils
### Model selection script
This script can be used to extensively train the best architectures found during a POPNAS search or a custom cell specification.
It uses a configuration file (by default _model_selection_training.json_), which is a subset of the search algorithm JSON file and its properties
override the ones used during search.

If the _params_ flag is set, the script also performs a pseudo-grid-search to tune the macro architectures of the selected cells, to generate additional
architectures to train which satisfy the provided parameters range (e.g. "2.5e6;3.5e6", must be semicolon separated).

Models and training procedure are slightly altered compared to search time: each model uses a secondary exit on the cell placed at 2/3 of total
model length, plus label smoothing is forced in loss. If using the default config, scheduled drop path and cutout are also forced to improve
the generalization during the extended number of epochs.

The script can be launched with the following command:
```
python scripts/model_selection_training.py -p {path_to_log_folder} -j {path_to_config_file}
```

### Last training script
This script is designed for training the best architecture found during the model selection, posterior to the NAS run.
It uses a configuration file (by default _last_training.json_), which is a subset of the search algorithm JSON file and its properties
override the ones used during search.

The model and training procedure are tweaked as done during model selection, but in this case the model is training on both
the training and validation data, without reserving a validation set.
It is important to check that the model generalizes well during the model selection phase, to not incur in overfitting.

Here, multiple command line arguments must be provided, to build the exact model without altering the configuration file:
- **-b**, the desired batch size
- **-f**, the model starting filters
- **-m**, the number of motifs used in the model
- **-n**, the number of normal cells to stack per motif
- **-spec**, the cell specification, in format "[(block0);(block1);...]"

The script can be launched with the following command (change parameters accordingly):
```
python scripts/last_training.py -p {path_to_log_folder} -j {path_to_config_file} -b 256 -f 24 -m 3 -n 2 -spec "[(-2, '1x3-3x1 conv', -1, '1x5-5x1 conv');(-1, '1x3-3x1 conv', 0, '2x2 maxpool')]"
```

### Evaluation script
Using this script, it's possible to evaluate a previously trained model on the test set of the dataset selected in the configuration.
The information is saved in the _eval.txt_ file inside the model folder, together with the confusion matrix.

The script can be launched with:
```
python scripts/evalaute_network.py -p {path_to_log_folder} -f {model_folder}
```


### Time prediction testing script
An additional script is provided to analyze the results of multiple time predictors, on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_time_test*) inside the log folder given as argument.

The script can be launched with the following command:
```
python scripts/predictors_time_testing.py -p {path_to_log_folder}
```

### Accuracy predictor testing script
Another additional script is provided to analyze the results of multiple accuracy predictor configurations, on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_acc_test*) inside the log folder given as argument.

The script can be launched with the following command:
```
python scripts/predictors_acc_testing.py -p {path_to_log_folder}
```

### Plot slideshow script
The plot_slideshow.py script is provided to facilitate visualizing related plots in an easier and faster way.
To use it, only the log folder must be provided.

An example of the command usage (from src folder):
```
python ./scripts/plot_slideshow.py -p {path_to_log_folder}
```
Close a plot overview to visualize the next one, the program terminates after showing all plots.

If the **--save** flag is specified, it will instead save all slides into 'plot_slides' folder, inside the log folder provided as -p argument.

If regressor_testing and/or controller_testing scripts have been run on data contained in selected log folder,
their plots will be appended in additional slides.


### Tensorboard
Trained CNNs have a callback for saving info to tensorboard log files. To access the training data associated to all
neural networks sampled during the search process, run the command:
```
tensorboard --logdir {path_to_log_folder}/tensorboard_cnn
```
In each tensorboard folder, there are also some additional files like the model summary and schema,
to have a quick overview of its structure.


### NAS-Bench-201
The _run_bench.py_ script defines a run configuration and extra utilities to map POPNAS architectures into NAS-Bench-201 genotype.

POPNAS use the latest API provided by NATS-bench (topology search space is equivalent to NAS-Bench-201), but requires to download the bench files
(see the instructions at: https://github.com/D-X-Y/NATS-Bench/blob/main/README.md, download the tss bench file).

The experiment can be run with command:
```
python ./run_bench.py -p {path_to_folder_with_NATS_bench_files}
```