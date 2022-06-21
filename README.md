# POPNASv2
Second version of POPNAS algorithm, a neural architecture search method developed for a master thesis by Matteo Vantadori
(Politecnico di Milano, academic year 2018-2019), based on 
[PNAS paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf).
This new version improves the time efficiency of the search algorithm, but also drastically increase the accuracy of the networks found,
making it competitive with other NAS works. It also fixes problems and bugs of the original version. 

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
docker run -it --rm -v %cd%:/exp --name popnas falanti/popnas:tf2.7.0gpu python run.py -j configs/run_debug.json --cpu
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
- **max_children**: defines the maximum amount of cells the algorithm can train in each iteration
  (except the first step, which trains all possible cells).
- **max_exploration_children**: defines the maximum amount of cells the algorithm can train in the exploration step.
- **lookback_depth**: maximum lookback depth to use. Lookback inputs are associated to previous cells,
  where _-1_ refers to last generated cell, _-2_ a skip connection to second-to-last cell, etc... 
- **lookforward_depth**: maximum lookforward depth to use. TODO: actually not supported, should always be null.
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


## Changelog from original version

### Generated CNN structure changes
- Fix cell structure to be an actual DAG, before only flat cells were generated (it was not possible to use other blocks output
  as input of another block).
- Fix blocks not having addition of the two operations output.
- Fix skip connections (input index -2) not working as expected.
- Tweak child CNN training hyperparameters, to make them more similar to PNAS paper.
- Add AdamW as an alternative optimizer that can be optionally enabled and used instead of Adam + L2 regularization, providing a more
  accurate weight decay.
- Add ScheduledDropPath in local version (see FractalNet paper), since it was used in both PNAS and NASNet.
- Add cosine decay restart learning rate schedule, since also ADAM benefits a lot from it.
- Allow the usage of flexible kernel sizes for each operator supported by the algorithm, thanks to regex parsing.
- Add support for TransposeConvolution operator.
- Add the possibility to generate models with multiple outputs, one at the end of each cell. The CNN is then trained with custom loss weights
  for each output, scaling exponentially based on the cell index (last cell loss has weight 1/2, the second-last 1/4, the third-last 1/8, etc...)


### Equivalent networks detection
- Equivalent blocks are now excluded from the search space, like in PNAS.
- Equivalent models (cell with equivalent structure) are now pruned from search, improving pareto front quality.
  Equivalent models could be present multiple times in pareto front before this change, this should improve a bit the diversity of the models trained.


### Predictors changes
- CatBoost can now be used as time regressor, instead or together with aMLLibrary supported regressors. By default, CatBoost is now used in time
  predictions of each expansion step, with automatic hyperparameters tuning based on random search.
- Add back the input columns features to regressor, as now inputs really are from different cells/blocks, unlike original implementation.
  Therefore, input values have a great influence on actual training time and must be used by regressor for accurate estimations.
- Use a newer version of aMLLibrary, with support for hyperopt. POPNAS algorithm and the additional scripts also use aMLLibrary multithreading to
  train the models faster, with a default number of threads equal to accessible CPU cores (affinity mask).
- Keras predictors support hyperparameters tuning via Keras Tuner library. The search space for the parameters is
  defined in each class and can be overridden by each model to adapt it for each specific model structure.
- Add predictors testing scripts, useful to tune their hyperparameters on data of an already completed run (both time and accuracy).
  These scripts are useful to tune the predictors in case their results are far from optimal on the given dataset.
- A totally new feature set has been engineered for time predictors, based mainly the dag representation of a cell specification. This new set
  is quite small but very indicative of the main factors that dictates time differences among the networks. It also implicitly generalize on
  equivalent cell specifications, since they have the same DAG representation and so the same features, making futile the data augmentation step.
- Accuracy predictor is now an ensemble of 5 models, each one trained on 4/5 of the data.
  This closely follows PNAS work and improved slightly the results.

### Exploration step
- Add an exploration step to POPNAS algorithm. Some inputs and operators could not appear in pareto front
  networks (or appear very rarely) due to their early performance, making the predictors penalizing them heavily also
  in future steps. Since these inputs and operators could be actually quite effective in later cell expansions,
  the exploration step now trains a small set of networks that contains these underused values. It also helps to
  discover faster the value of input values >= 0, since they are unknown in B=1 step and progressively added in future steps.


### Data extrapolation and data analysis
- Add plotter module, to analyze csv data saved and automatically producing relevant plots and metrics while running the algorithm.
- Add the plot slideshow script to visualize all produced plots easily in aggregated views.
- Add _avg_training_time.csv_ to automatically extrapolate the average CNN training time for each considered block size.
- Add _multi_output.csv_ to extrapolate the best accuracy reached for each cell output, when _multi_output_ is enabled in JSON config.


### Software improvements and refactors
- Migrate code to Tensorflow 2, in particular to the 2.7 version.
- The algorithm now supports 4 different image classification datasets: CIFAR10, CIFAR100, fashionMNIST and EuroSAT. It should be easy to implement
  support for other datasets and datasets supported by [Tensorflow-datasets](https://www.tensorflow.org/datasets/catalog/overview?hl=en)
  could work fine without any code change.
- Now use JSON configuration files, with some optional command line arguments. This approach is much more flexible and makes easier to parametrize
  all the various components of the algorithm run. Many parameters and hyperparameters that were hardcoded in POPNAS initial version are now
  tunable from the JSON config.
- Implement an actually working run restoring functionality. This allows to resume a previously interrupted run.
- CNN training has been refactored to use Keras model.fit method, instead of using a custom tape gradient method.
  New training method supports data augmentation and allows the usage of weight regularization if parameter is provided.
- LSTM controller has been refactored to use Keras API, instead of using a custom tape gradient method.
  This make the whole procedure easier to interpret and also more flexible to further changes and additions.
- Add predictors hierarchy (see _predictors_ folder). Predictor abstract class provides a standardized interface for all regressor methods
  tested during the work. Predictors can be either based on ML or NN techniques, they just need to satisfy the interface to be used during POPNAS
  algorithm and the additional scripts.
- Encoder has been totally refactored since it was a total mess, causing also a lot of confusion inside the other modules.
  Now the state space stores each cell specification as a list of tuples, where the tuples are the blocks (input1, op1, input2, op2).
  The encoder class instead provides methods to encode/decode the inputs and operators values, with the possibility of adding multiple encoders
  at runtime and using them easily when needed. The default encoders are now 1-indexed categorical integers, instead of the 0-indexed used before. 
- Improve immensely virtual environment creation, by using _Poetry_ tool to easily install all dependencies.
- Improve logging (see log_service.py), using standard python log to print on both console and file. Before, text logs were printed only on console.
- Implement saving of best model, so that can be easily trained after POPNAS run for further experiments.
- A script is provided to train the best model from checkpoint saved during the POPNAS run. It can also recreate the best network from scratch or
  train a custom cell specification, provided as argument.
- Format code with pep8 and flake, to follow standard python formatting conventions.
- General code fixes and improvements, especially improve readability of various code parts for better future maintainability.
  Many blob functions have been finely subdivided in multiple sub-functions and are now properly commented and typed.
  Right now almost the total codebase of original POPNAS version have been refactored, either due to structural or quality changes.


### Command line arguments changes
- Add --cpu option to easily choose between running on CPU or GPU.
- Add --pnas option to run without regressor, making the procedure similar to original PNAS algorithm.
- Add a lot of new configuration parameters, which can be set in the new JSON configuration file.


### Other bug fixes
- Fix regressor features bug: dynamic reindexing was inaccurate due to an int cast that was totally unnecessary since the dynamic reindex is
  designed to be a float value.
- Fix training batch processing not working as expected, last batch of training of each epoch could have contained duplicate images
  due to how repeat was wrongly used before batching.
- Fix tqdm bars for model predictions procedure, to visualize better its progress.



## TODO
- Improve the restoring function and investigate potential bugs (especially in prediction and expansion phase it could not work properly, since
  I only wrote the logic to stop it during CNN training, which should be the 90% of the cases).
