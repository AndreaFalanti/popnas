# POPNASv2
Second version of POPNAS algorithm, a neural architecture search method based on 
[PNAS paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf).
The first version has been developed for a master thesis by Matteo Vantadori (Politecnico di Milano, academic year 2018-2019).

This new version improves the time efficiency of the search algorithm, passing from an average 2x speed-up to
an average 4x speed-up compared to PNAS on same experiment configurations.

The accuracy of the final neural network architectures found are now competitive with other NAS works,
solving the main drawback of the first POPNAS version, which had a large accuracy GAP with PNAS and similar methods. 

## Installation
This section provides information for installing all needed software and packages for properly run POPNASv2 on your system. If you prefer, you can
use the provider Docker container and avoid these steps (see Docker section below).

### Required additional software and tools
Virtual environment and dependencies are managed by _poetry_, check out its [repository](https://github.com/python-poetry/poetry)
for installing it on your machine and learning more about it.

You need to have installed either python version 3.7 or 3.8 for building a valid environment (python versions > 3.8 could work,
but they have not been officially tested).
To install and manage the python versions and work with _poetry_ tool, it's advised to use [pyenv](https://github.com/pyenv/pyenv)
or [pyenv-win](https://github.com/pyenv-win/pyenv-win) based on your system.

Make also sure that *graphviz* is installed in your machine, since it is required to generate plots of Keras models.
Follow the installation instructions at: https://graphviz.gitlab.io/download/.

### Installation steps
After installing poetry, optionally run on your terminal:
```
poetry config virtualenvs.in-project true
```
to change the virtual environment generation position directly in the project folder, if you prefer so.

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

After that you should be able to run POPNASv2 with this command:
```
python run.py
```

### GPU support
To use a GPU locally, you must satisfy Tensorflow GPU hardware and software requirements.
Follow https://www.tensorflow.org/install/gpu instructions to setup your device, make sure
to install the correct versions of CUDA and CUDNN for Tensorflow 2.7 (see https://www.tensorflow.org/install/source#linux).

## Build Docker container
In _docker_ folder it's provided a dockerfile to extend an official Tensorflow container with project required pip packages
and mount POPNAS source code.

To build the image, open the terminal into the _src_ folder and execute this command:
```
docker build -f ../docker/Dockerfile -t falanti/popnas:tf2.7.0gpu .
```

POPNASv2 can then be launched with command (set arguments as you like):
```
docker run -it --rm -v %cd%:/exp --name popnas falanti/popnas:tf2.7.0gpu python run.py -j configs/run_debug.json
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

Here it's presented a list of the configuration sections and fields, with a brief description.

**Search Space**:
- **blocks**: defines the maximum amount of blocks a cell can contain.
- **lookback_depth**: maximum lookback depth to use (in absolute value). Lookback inputs are associated to previous cells,
  where _-1_ refers to last generated cell, _-2_ a skip connection to second-to-last cell, etc...
- **operators**: list of operators that can be used inside each cell. Note that the string format is important,
  since they are recognized by regexes.
  Actually supported operators, with customizable kernel size(@):
  - identity
  - @x@ dconv      (Depthwise-separable convolution)
  - @x@-@x@ conv
  - @x@ conv
  - @x@ maxpool
  - @x@ avgpool
  - @x@ tconv      (Transpose convolution)

**Search strategy**:
- **max_children**: defines the maximum amount of cells the algorithm can train in each iteration
  (except the first step, which trains all possible cells).
- **max_exploration_children**: defines the maximum amount of cells the algorithm can train in the exploration step.
- **pareto_objectives**: defines the objectives considered during the search for optimizing the selection of the neural network architectures to
  train. Currently supported values are [accuracy, time, params], at least two of them must be provided and accuracy must always be set to have
  meaningful results.
  
**CNN hyperparameters**:
- **epochs**: defines for how many epochs E each child network has to be trained.
- **learning_rate**: defines the learning rate of the child CNN networks.
- **filters**: defines the initial number of filters to use, which increase in each reduction cell.
- **weight_reg**: defines the L2 regularization factor to use in CNNs. If _null_, regularization is not applied.
- **use_adamW**: use adamW instead of standard L2 regularization.
- **drop_path_prob**: defines the max probability of dropping a path in _scheduled drop path_. If set to 0,
  then _scheduled drop path_ is not used.
- **cosine_decay_restart**: dictionary for hyperparameters about cosine decay restart schedule.
  - **enabled**: if _true_ use cosine decay restart, _false_ instead for using a plain learning rate schedule
  - **period_in_epochs**: first decay period in epochs, changes at each period based on _m_mul_ value.
  - **[t_mul, m_mul, alpha]**:
    see [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts).
- **softmax_dropout**: probability of dropping a value in output Softmax layer. If set to 0, then dropout is not used.
  Note that dropout is used on each output when _multi_output_ flag is set.


**CNN architecture parameters**:
- **motifs**: number of motifs to stack in each CNN. In NAS literature, a motif usually refers to a single cell,
  here instead it is used to indicate a stack of N normal cells followed by a single reduction cell.
- **normal_cells_per_motif**: normal cells to stack in each motif.
- **concat_only_unused_blocks**: if _true_, only blocks' output not used internally by the cell
  will be used in final concatenation cell output, following PNAS and NASNet.
  If set to _false_, all blocks' output will be concatenated in final cell output.
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
- **name**: used to identify and load a Keras or TFDS dataset supported by POPNAS.
  Can be _null_ if the path of a custom dataset is provided.
- **path**: path to a folder containing a custom dataset.
  Can be _null_ if you want to use a supported dataset already present in Keras or TFDS.
- **classes_count**: classes present in the dataset. If using a Keras dataset, this value can be inferred automatically.
- **batch_size**: defines the batch size dimension of the dataset.
- **inference_batch_size**: defines the batch size dimension for benchmarking the inference time of a network.
- **validation_size**: fraction of the total samples to use for validation set, e.g. _0.1_ value means that 10% of the
  training samples will be reserved for the validation dataset. Can be _null_ for TFDS dataset which have already
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
- **resize**: dictionary with parameters related to image resizing.
  - **enabled**: _true_ for using resizing, _false_ otherwise.
  - **width**: target image width in pixels.
  - **height**: target image height in pixels.
- **data_augmentation**: dictionary with parameters related to data augmentation.
  - **enabled**: _true_ for using data augmentation, _false_ otherwise.
  - **perform_on_gpu**: perform data augmentation directly on GPU (through Keras experimental layers).
    Usually advised only if CPU is very slow, since CPU prepares the images while the GPU trains the network
    (asynchronous prefetch), instead performing data augmentation on the GPU will make the process sequential,
    always causing delays even if it's faster to perform on GPU.

**Others**:
- **accuracy_predictor_ensemble_units**: defines the number of models used in the accuracy predictor (ensemble).
- **predictions_batch_size**: defines the batch size used when performing both time and accuracy predictions in controller
  update step (predictions about cell expansions for blocks b+1). Incrementing it should decrease the prediction time
  linearly, up to a certain point, defined by hardware resources used.
- **save_children_weights**: if _true_, best weights of each child neural network are saved in log folder.
- **save_children_as_onnx**: if _true_, each child neural network will be serialized and saved as ONNX format.
- **pnas_mode**: if _true_, the algorithm will not use most of the improvements introduced by POPNAS, mainly the
  temporal regressor, Pareto optimality and exploration step, making the search process very similar to PNAS.
- **train_strategy**: defines the type of device and distribution strategy used for training the architectures sampled by the algorithm.
  Currently supports only local training with a single device. Accepted values: [CPU, GPU, TPU].


## Output folder structure
Each run produce a single output folder, which contains all the files related to the run results.
Some files are generated only if the related configuration flag is set to true, refer to JSON configuration file.

The files are organized in different subfolders:
- **best_model**: contains the checkpoint of the best model found during the search process.
- **csv**: contains many csv files with data extrapolated during the run, like the predictions and the training results.
- **plots**: contains all the plots automatically generated by the algorithm.
- **predictors**: contains logs and results about the predictors training process.
- **restore**: contains additional information for restoring an interrupted run. Contains also the input configuration file.
- **tensorboard_cnn**: contains tensorboard logs, model structure and summary of each neural network trained by the algorithm.


## Additional scripts and utils
### Final training script
This script can be used to train the best architecture found by a run or a custom cell specification easily, for
more extensive training tests. It use a configuration file, which is a subset of the search algorithm JSON file
(it has only the sections _cnn_hp_, _architecture_parameters_, _dataset_).

The script can be launched with the following command:
```
python scripts/final_training.py -p {path_to_log_folder} -j {path_to_config_file}
```


### Time prediction testing script
An additional script is provided to analyze the results of multiple predictors on time target, on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_time_test*) inside the log folder given as argument.

The script can be launched with the following command:
```
python scripts/predictors_time_testing.py -p {path_to_log_folder}
```

### Controller testing script
Another additional script is provided to analyze the results of multiple controller configurations on the data gathered in a POPNAS run.
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

If **--save** flag is specified, it will instead save all slides into 'plot_slides' folder, inside the log folder provided as -p argument.

If regressor_testing and/or controller_testing scripts have been run on data contained in selected log folder,
also their output plots will be visualized in additional slides at the end.


### Tensorboard
Trained CNNs have a callback for saving info to tensorboard log files. To access the training data associated to all
neural networks sampled during the search process, run the command:
```
tensorboard --logdir {path_to_log_folder}/tensorboard_cnn --port 6096
```
In each tensorboard folder it's also present the model summary as txt file, to have a quick and simple overview of its structure.
