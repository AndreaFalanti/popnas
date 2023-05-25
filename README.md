# POPNASv3

Third version of the **Pareto-Optimal Progressive Neural Architecture Search (POPNAS)** algorithm, a neural architecture search method based on 
[PNAS](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf), and direct extension
and improvement of the second version.

**POPNASv2** has been developed by Andrea Falanti for his master's thesis at Politecnico di Milano, and the work has also been
published at the IEEE IJCNN 2022.
The paper is available at: https://ieeexplore.ieee.org/abstract/document/9892073.
The second version improves the time efficiency of the search algorithm, passing from an average 2x speed-up to
an average 4x speed-up compared to PNAS on the same experiment configurations.
The top-accuracy neural network architectures found are competitive with PNAS and other state-of-the-art methods,
solving the main drawback of the first POPNAS version.

**POPNASv3** extends the second version by addressing time series classification problems, adding new operators like LSTM, GRU
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

Minor version updates performed after version 3 focus on software refactors to make POPNAS extendable to other supervised tasks, designing
specific class hierarchies to handle each aspect of the process and make them customizable based on the task addressed.
In particular, it is possible to implement custom behaviors for _DatasetGenerator_, _ModelGenerator_, and _OpAllocators_ to respectively
load different data formats, use custom macro-architectures for the implemented models, and define possible
additional operators relevant for the considered task. POPNASv3 main functional updates include the support for semantic segmentation tasks,
change of the accuracy and time predictor models to GIN and SVR, respectively, support for XLA and mixed precision,
and direct optimization of the inference time in the Pareto front.

## Citations
**POPNASv2:**
```
@inproceedings{falanti2022popnasv2,
  title={Popnasv2: An efficient multi-objective neural architecture search technique},
  author={Falanti, Andrea and Lomurno, Eugenio and Samele, Stefano and Ardagna, Danilo and Matteucci, Matteo},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```

**POPNASv3:**
```
@article{falanti2022popnasv3,
  title={POPNASv3: a Pareto-Optimal Neural Architecture Search Solution for Image and Time Series Classification},
  author={Falanti, Andrea and Lomurno, Eugenio and Ardagna, Danilo and Matteucci, Matteo},
  journal={arXiv preprint arXiv:2212.06735},
  year={2022}
}
```

## Installation
This section provides information for installing all needed software and packages for properly run POPNASv3 on your system. If you prefer, you can
use the provided Docker container and skip these steps (see the Docker section below).

### Required additional software and tools
Virtual environment and dependencies are managed by _poetry_, check out its [repository](https://github.com/python-poetry/poetry)
for installing it on your machine and learning more about it.
Version >= 1.2.0 is required.

The supported python versions are listed in the _pyproject.toml_ file, in the first line of the dependencies section.
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
pyenv install 3.10.9 # or any valid version
pyenv shell 3.10.9
```

To install the dependencies, simply run:
```
poetry install
```
Poetry will generate a virtual env based on the active python version and install all the packages there.

If poetry does not automatically select your active python version (or a valid one), use the following command to generate the environment:
```
poetry env use /full/path/to/python
```
and repeat the `poetry install` command.

You can activate the new environment with command:
```
poetry shell
```

The _aMLLibrary_ dependency is integrated in the project as a git submodule, since it is not possible to install it with package managers.
_aMLLibrary_ dependencies are handled by Poetry, but the library must be downloaded separately with the command:
```
git submodule update --init --recursive
```

After that, you should be able to run POPNASv3 with this command:
```
python run.py
```

### GPU support
To enable GPU computations locally, you must satisfy Tensorflow GPU hardware and software requirements.
Follow https://www.tensorflow.org/install/gpu instructions to set up your device. Make sure
to install the exact versions of CUDA and CUDNN for Tensorflow 2.10 (see https://www.tensorflow.org/install/source#gpu).

### Note for fellow developers
If you open the project with an IDE, the imports from _tensorflow.keras_ will be marked as errors and autocomplete will not work.
Actually, the imports work fine, but the Keras library is lazily loaded by tensorflow, causing 
[issues](https://github.com/tensorflow/tensorflow/issues/56231) with autocomplete.

You can solve the problem by creating a symlink to Keras inside the Tensorflow package located in the virtual environment,
using the following command:
```
mklink /D keras ..\keras\api\_v2\keras      (windows)
ln -s ..\keras\api\_v2\keras keras          (linux)
```

## Build Docker container
In the _docker_ folder, it is provided a dockerfile to extend an official Tensorflow container with pip packages required by the project
and finally mounting POPNAS source code.

To build the image, open the terminal into the root folder and execute this command:
```
docker build -f docker/Dockerfile -t andreafalanti/popnas:v3 .
```

POPNASv3 can then be launched with command (set arguments as you like):
```
docker run -it --name popnas andreafalanti/popnas:v3 python run.py -j configs/run_debug.json
```

## Run configuration
### Command line arguments
All command line arguments are optional.
- `-j`: specifies the path of the json configuration to use. If non provided, _configs/run.json_ will be used.
- `-r`: used to restore a previous interrupted run. Specifies the path of the log folder of the run to resume.
- `--name`: specifies a custom name for the log folder. If not provided, it defaults to date-time
  in which the run is started.

### Json configuration file
The run behaviour can be customized through the usage of custom json files. By default, the _run.json_ file
inside the _configs_ folder will be used. This file can be used as a template and customized to generate new configurations.
A properly structured json config file can be provided to the algorithm by specifying its path in `-j` command line argument.

Here it is presented a list of the configuration sections and fields, with a brief description.

**Search Space**:
- `blocks`: defines the maximum number of blocks a cell can contain.
- `lookback_depth`: maximum lookback depth to use (in absolute value). Lookback inputs are associated to previous cells,
  where _-1_ refers to last generated cell, _-2_ a skip connection to second-to-last cell, etc...
- `operators`: list of operators that can be used inside each cell. Note that the string format is important,
  since they are recognized by regexes.
  The currently supported operators, with customizable integer parameters(@) for kernel size and other parameters based on the operation type, are:
  - identity
  - @x@:@dr dconv (depthwise-separable convolution)
  - @x@-@x@ conv (spatial-separable convolutions)
  - @x@:@dr conv
  - @x@:@z zconv (zoomed convolution, see FasterSeg paper)
  - @x@ maxpool
  - @x@ avgpool
  - @x@ tconv (transpose convolution)
  - @r SE (Squeeze and Excitation)
  - (2D only) @k-@h-@b cvt (Convolutional Vision Transformer)
  - (2D only) @k-@h scvt (Simplified Convolutional Vision Transformer, custom operator not from literature)
  - (1D only) lstm
  - (1D only) gru

  conv and dconv support an optional group _:@dr_ for setting the dilation rate, which can be omitted to use non-dilated convolutions.

  For time series (1D inputs), specify the kernel size as @ instead of @x@, since the kernel size is mono dimensional.

**Search strategy**:
- `max_children`: defines the maximum number of cells the algorithm can train in each iteration
  (except the first step, which trains all possible cells composed by a single block).
- `max_exploration_children`: defines the maximum number of cells the algorithm can train in the exploration step.
- `score_metric`: specifies the metric used for estimating the prediction quality of the trained models.
  Currently supported: [`accuracy`, `f1_score`, `mean_iou`].
  - `additional_pareto_objectives`: defines the additional objectives considered during the search alongside the score metric, for optimizing
    the selection of the neural network architectures to train.
    Currently supported values: [`time`, `params`, `inference_time`]. POPNAS requires at least one of them.
  
**Training hyperparameters**:
- `epochs`: defines for how many epochs each child network has to be trained.
- `learning_rate`: defines the learning rate of the child networks.
- `weight_decay`: defines the weight decay to apply. For optimizers not supporting weight decay,
  L2 regularization is instead applied to all the weights of the model. If _null_, regularization is not applied.
- `drop_path`: defines the max probability of dropping a path in _scheduled drop path_. If set to 0,
  then _scheduled drop path_ is not used.
- `softmax_dropout`: probability of dropping a value in output Softmax layer. If set to 0, then dropout is not used.
  Note that dropout is used on each output when _multi_output_ flag is set.
- `optimizer`: dictionary for hyperparameters about the optimizer definition. See the TensorFlow documentation for details
  about the optimizers and schedulers hyperparameters.
  - `type`: string defining the optimizer class and potential hyperparameters. Supported formats: [`adamW`, `adam`,
    `SGDW: \f momentum`, `SGD: \f momentum`, `radam: \f alpha`], with `\f` representing a float number.
    Parameters are optional, so you can just specify the name (e.g., "SGD") and default values will be used.
  - `scheduler`: string defining the optimizer class and potential hyperparameters. Supported formats: 
    [(CosineDecayRestart) `"cdr: \f period, \f t_mul, \f m_mul, \f alpha"`,
    (CosineDecay) `"cd: \f alpha"`], with `\f` representing a float number.
    Parameters are optional, so you can just specify the name (e.g., "cdr") or a subset of parameters,
    and default values will be used for undefined groups. NOTE: parameters must be ordered as in specification.
  - `warmup`: optional number of epochs where the learning rate scales linearly from 0 to the target learning rate.
    After warmup, the specified scheduler is applied. Warmup can stabilize training results.
  - `lookahead`: optional object which can enable the lookahead mechanism when defined.
    - `sync_period`: integer defining the number of steps before syncing slow weights.
    - `slow_step_size`: float value indicating the ratio for updating the slow weights.
- `label_smoothing`: float between 0 and 1, it can be applied in CategoricalCrossentropy loss of classification tasks.
  Ignored in segmentation tasks, since unsupported on sparse labels.


**Architecture hyperparameters**:
- `filters`: defines the initial number of filters to use, which increase in each reduction cell.
- `motifs`: number of motifs to stack in each neural network. In NAS literature, a motif usually refers to a single cell,
  here instead it is used to indicate a stack of N normal cells followed by a single reduction cell.
- `normal_cells_per_motif`: normal cells to stack in each motif.
- `block_join_operator`: defines the operator used to join the tensors produced by the two operators of a block.
  Supported values: [`add`, `avg`].
- `lookback_reshape`: if _true_, when a skip lookback (-2) has a different shape from the one expected by the current cell, it is reshaped
  with a pointwise convolution, before passing it to block operators (as done in PNAS). If _false_, the skip lookbacks are instead
  passed directly to block operators requesting them, which would operate as reduction cell case when the shape diverges from expected one.
- `concat_only_unused_blocks`: if _true_, only blocks' output not used internally by the cell
  will be used in final concatenation cell output, following PNAS and NASNet.
  If set to _false_, all blocks' output will be concatenated in final cell output.
- `residual_cells`: if _true_, each cell output will be followed by a sum with the nearest lookback input used, to make the output residual
  (see ResNet paper). If lookback and cell output shapes diverge, pointwise convolution and/or max pooling are performed to adapt 
  the spatial dimension and the filters.
- `se_cell_output`: if _true_, squeeze and excitation is performed on the cell output (before residual, if used together).
- `multi_output`: if _true_, sampled networks will have an output exit (GAP + Softmax) at the end of each cell.
- `activation_function`: string name of the Keras activation function to use in operators.
  If not provided, it defaults to Swish.

[//]: # (**RNN hyperparameters &#40;controller, optional&#41;**: \\)

[//]: # (If the parameters are not provided or the object is omitted in JSON config, default parameters will be applied.)

[//]: # (The accepted parameters and their names depend on the model type chosen for the controller.)

[//]: # (See KerasPredictor subclasses to have a better idea &#40;TODO&#41;.)

[//]: # (- `epochs`: how many epochs the LSTM is trained on, at each expansion step.)

[//]: # (- `lr`: LSTM learning rate.)

[//]: # (- `wr`: LSTM L2 weight regularization factor. If _null_, regularization is not applied.)

[//]: # (- `er`: LSTM L2 weight regularization factor applied on embeddings only. If _null_, regularization is not applied.)

[//]: # (- `embedding_dim`: LSTM embedding dimension, used for both inputs and operator embeddings.)

[//]: # (- `cells`: total LSTM cells of the model.)

**Dataset**:
- `type`: specifies the problem domain of the provided data. Processing pipeline and other parameters depend on the
  addressed task. Supported values: [`image_classification`, `time_series_classification`, `image_segmentation`].
- `name`: used to identify and load a Keras or TFDS dataset supported by POPNAS.
  Can be _null_ if the path of a custom dataset is provided.
- `path`: path to a folder containing a custom dataset.
  Can be _null_ if you want to use a supported dataset already present in Keras or TFDS.
- `classes_count`: classes present in the dataset. If using a Keras dataset, this value can be inferred automatically.
- `ignore_class`: optional parameter used only in image segmentation problems. Defines the integer value of a class
  (e.g., background or void class) to ignore during loss and mean IoU computations (accuracy will still consider it,
  no current support for masking in TF).
- `batch_size`: defines the batch size used for the training split.
- `val_test_batch_size`: defines the batch size used for the validation and test splits. If not provided, it will default to _batch_size_.
- `inference_batch_size`: defines the batch size dimension for benchmarking the inference time of a network. Defaults to 1.
- `validation_size`: fraction of the total samples to use for the validation set, e.g. _0.1_ value means that 10% of the
  training samples will be reserved for the validation dataset. Can be _null_ for TFDS dataset which have
  a separated validation set, for using it instead of partitioning the training set.
- `cache`: if _true_, the dataset will be cached in memory, increasing the training performance.
  Strongly advised for small datasets, but also for big ones if they can fit in the available RAM.
- `folds`: number of dataset folds to use. When using multiple folds, the metrics extrapolated from network training
  will be the average of the ones obtained on each fold.
- `samples`: if provided, limits the total dataset samples used by the algorithm to the number provided (integer).
  Useful for fast testing.
- `balance_class_losses`: if _true_, the class losses will be weighted proportionally to the number of samples.
  The exact formula for computing the weight of each class is:
  w<sub>class</sub> = 1 / (classes_count * samples_fraction<sub>class</sub>).
- `class_labels_remapping`: optional dictionary to remap class labels. The keys are strings representing the original class labels,
  the value associated with each key is instead an integer, representing the new label to assign to that class
  (e.g., {"1": 0, "2": 255} will convert labels=1 to 0 and labels=2 to 255).
- `data_augmentation`: dictionary with parameters related to data augmentation.
  - `enabled`: _true_ for using data augmentation, _false_ otherwise.
- ...extra parameters depending on dataset type, see next sections.

**Dataset (_image_classification_ or _image_segmentation_)**:
- `resize`: an optional object with parameters related to image resizing. Applied only if the object is defined.
  In _image segmentation_ tasks, the given values represent the size of the random crop applied at training time.
  - `width`: target image width in pixels.
  - `height`: target image height in pixels.

**Dataset (_time_series_classification_ only)**:
- `rescale`: if _true_ the values will be rescaled with a factor based on 98 percentile of the entire input values.
- `normalize`: if _true_, sample values will be shifted and scaled into a distribution centered around 0 with standard deviation 1 (z-normalization).

**Others**:
- `accuracy_predictor_ensemble_units`: defines the number of models used in the accuracy predictor (ensemble).
- `predictions_batch_size`: defines the batch size used by predictors during the controller
  update step (estimates about cell expansions of blocks b+1).
  Incrementing it should decrease the prediction time linearly, up to a certain point, defined by hardware resources used.
- `save_children_weights`: if _true_, the best weights of each sampled neural network will be saved in log folder.
- `save_children_models`: if _true_, each sampled neural network will be serialized and saved both as ONNX and TF format.
- `pnas_mode`: if _true_, the algorithm will not use most of the improvements introduced by POPNAS, mainly the
  temporal regressor, Pareto optimality and exploration step, making the search process very similar to PNAS.
- `train_strategy`: defines the type of device and distribution strategy used for training the architectures sampled by the algorithm.
  Currently, it supports only local training with a single device. Supported values: [`CPU`, `GPU`, `multi-GPU`, `TPU`].
- `enable_XLA_compilation`: enable XLA compilation when compiling Keras models.
  It is incompatible with some operators (e.g., bilinear upsample), and cause memory leaks on prolonged experiments, use it with caution.
- `use_mixed_precision`: use float16 when performing computations inside Keras models.
  Should improve performances on GPUs equipped with tensor cores.
  Also, it makes it possible to increase the batch size due to lower memory consumption.


## Output folder structure
Each run produces a single output folder, which contains all the files related to the run results.
Some files are generated only if the related configuration flag is set to true, refer to the JSON configuration file.

The files are organized in different subfolders:
- **best_model**: contains the checkpoint of the best model found during the search process.
- **csv**: contains many csv files with data extrapolated during the run, like the predictions and the training results.
- **plots**: contains all the plots automatically generated by the algorithm.
- **predictors**: contains logs and results about the predictors training process.
- **restore**: contains additional information for restoring an interrupted run. Contains also the input configuration file.
- **sampled_models**: contains TensorBoard logs, model structure and summary of each neural network trained by the algorithm at search time.


## Additional scripts and utils
The _src/scripts_ folder contains additional scripts that can be used to further analyze and process the neural networks found
during the POPNAS search procedure.
Refer to the README placed in the _scripts_ folder for additional information.

### Logging runs on Neptune
If you have an account on [neptune.ai](https://neptune.ai/), you can easily log any POPNAS experiment to your Neptune workspace.
POPNAS checks for the existence of two environment variables on your system:
- `NEPTUNE_API_TOKEN`: set it to your personal API token
- `NEPTUNE_WORKSPACE`: set it to the target workspace name

If both are provided, POPNAS will create a new Neptune project and store the results of all models in separate runs.
When using Docker, you can provide the environment variables with the `-e` flag, or `--env-file` flag if you prefer to store them in a file.


### Tensorboard
Trained neural networks have a callback for saving info into TensorBoard log files.
To access the TensorBoard training logs of all neural networks sampled during the search process, run the command:
```
tensorboard --logdir {path_to_log_folder}/sampled_models
```
In each model folder, there are also some additional files like the model summary, the architecture graph schema,
and the ONNX file if the run was configured to save all models sampled during the search.


### NAS-Bench-201
The _run_bench.py_ script defines a run configuration and extra utilities to map POPNAS architectures into NAS-Bench-201 genotype.

POPNAS use the latest API provided by NATS-bench (topology search space is equivalent to NAS-Bench-201), but requires to download the bench files
(see the instructions at: https://github.com/D-X-Y/NATS-Bench/blob/main/README.md, download the tss bench file).

The experiment can be run with command:
```
python run_bench.py -p {path_to_folder_with_NATS_bench_files}
```

### Flask server
A flask server is provided to execute POPNAS processes from external remote interfaces.
The server is mainly used for the web deployment, for architectures using a single POPNAS docker container.

The Flask server can be run with the command:
```
flask --app "server/app.py:app" run
```