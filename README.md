# POPNASv2
Fix and refactor of POPNAS, a neural architecture search method developed for a master thesis by Matteo Vantadori (Politecnico di Milano, academic year 2018-2019),
based on [PNAS paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf).

## Installation
This section provides information for installing all needed software and packages for properly run POPNASv2 on your system. If you prefer, you can
use the provider Docker container and avoid these steps (see Docker section below).

### Required additional software and tools
Virtual environment and dependencies are managed by *poetry*, check out its [repository](https://github.com/python-poetry/poetry)
for installing it on your machine.

You need to have installed python version 3.6.9 or 3.7.4 (advised for windows) for building a valid environment (other versions could work, but are not tested).
To install and manage the python versions and work with *poetry* tool, it's advised to use [pyenv](https://github.com/pyenv/pyenv)
or [pyenv-win](https://github.com/pyenv-win/pyenv-win) based on your system.

Make sure also that *graphviz* is installed in your machine, since it is required to generate plots of keras models.
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
pyenv install 3.6.9 # or 3.7.4
pyenv shell 3.6.9
```

To install the dependencies, simply execute:
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
python run.py -b 5 -k 256
```

### GPU support
To use GPU locally, you must satisfy Tensorflow GPU hardware and software requirements.
Follow https://www.tensorflow.org/install/gpu instructions to setup your device, make sure
to install the correct versions of CUDA and CUDNN for Tensorflow 2.5 (see https://www.tensorflow.org/install/source#linux).

## Build Docker container
In *docker* folder it's provided a dockerfile to extend an official Tensorflow container with project required pip packages
and mount POPNAS source code.
To build the image, open the terminal into the *src* folder and execute this command:
```
docker build -f ../docker/Dockerfile -t falanti/popnas:py3.8.10-tf2.7.0gpu .
```

POPNASv2 can then be launched with command (set arguments as you like):
```
docker run -it --rm -v %cd%:/exp --name popnas-tf2.7 falanti/popnas:py3.8.10-tf2.7.0gpu python run.py -j configs/run_debug.json --cpu
```

## Run configuration
### Command line arguments
All command line arguments are optional.
- **-j**: specifies the path of the json configuration to use. If non provided, _configs/run.json_ will be used.
- **-r**: used to restore a previous interrupted run. Specifies the path of the log folder of the run to resume.
- **--cpu**: if specified, the algorithm will use only the cpu, even if a gpu is actually available.
- Must be specified if the host machine has no gpu.
- **--pnas**: if specified, the algorithm will not use a regressor, disabling time estimation.
  This will make the computation extremely similar to PNAS algorithm.

### Json configuration file
The run behaviour can be customized through the usage of custom json files. By default, the _run.json_ file inside the _configs_ folder
will be used. This file can be used as a template and customized to generate new configurations. A properly structured json config file can be
used by the algorithm by specifying its path in -j command line arguments.

Here it's presented a list of the configuration sections and fields, with a brief description.

**Search Space**:
- **blocks**: defines the maximum amount of blocks a cell can contain.
- **max_children**: defines the amount of top-K cells the algorithm picks up to expand at the next iteration.
- **max_exploration_children**: defines the maximum amount of cells the algorithm can train in the exploration step.
- **lookback_depth**: maximum lookback depth to use.
- **lookforward_depth**: maximum lookforward depth to use. TODO: actually not supported, should always be null.
- **operators**: list of operators that can be used inside each cell. Note that the string format is important, since they are recognized by regexes.
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
- **batch_size**: defines the batch size dimension of the dataset.
- **learning_rate**: defines the learning rate of the child CNN networks.
- **filters**: defines the initial number of filters to use.
- **weight_reg**: defines the L2 regularization factor to use in CNNs. If _null_, regularization is not applied.
- **cosine_decay_restart**: dictionary for hyperparameters about cosine decay restart schedule.
  - **enabled**: use cosine decay restart or not (plain learning rate)
  - **period_in_epochs**: first decay period in epochs
  - **[t_mul, m_mul, alpha]**: see [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts)

**CNN architecture parameters**:
- **motifs**: motifs to stack in each CNN. In NAS literature, a motif usually refers to a single cell, here instead it is used to indicate
  a stack of N normal cells followed by a single reduction cell.
- **normal_cells_per_motif**: normal cells to stack in each motif.
- **concat_only_unused_blocks**: when _true_, only blocks' output not used internally by the cell will be used in final concatenation cell output,
  following PNAS and NASNet. If set to _false_, all blocks' output will be concatenated in final cell output.

**LSTM hyperparameters**:
- **epochs**: how many epochs the LSTM is trained on, at each expansion step.
- **lr**: LSTM learning rate.
- **wr**: LSTM L2 weight regularization factor. If _null_, regularization is not applied.
- **er**: LSTM L2 weight regularization factor applied on embeddings only. If _null_, regularization is not applied.
- **embedding_dim**: LSTM embedding dimension, used for both inputs and operator embeddings.
- **cells**: total LSTM cells of the model.

**Dataset**:
- **name**: used to identify and load a Keras dataset. Can be _null_ if the path of a custom dataset is provided.
- **path**: path to a folder containing a custom dataset. Can be _null_ if you want to use a dataset already present in Keras.
- **classes_count**: classes present in the dataset. If using a Keras dataset, this value can be inferred automatically.
- **folds**: number of dataset folds to use. When using multiple folds, the metrics extrapolated from CNN training will be the average of
  the ones obtained on each fold.
- **samples**: if provided, limits the total dataset samples to the number provided (integer). This means that the total training and validation
  samples will amount to this value (or less if the dataset has actually fewer samples than the value indicated). Useful for fast testing.
- **data_augmentation**: dictionary with parameters related to data augmentation
  - **enabled**: use data augmentation or not.
  - **perform_on_gpu**: perform data augmentation directly on GPU (through Keras experimental layers).
    Usually advised only if CPU is very slow, since CPU prepares the images while the GPU trains the network (asynchronous prefetch),
    instead performing data augmentation on the GPU will make the process sequential, always causing delays even if it's faster to perform.

**Others**:
- **pnas_mode**: if _true_, the algorithm will not use the temporal regressor and pareto front search, making the run very similar to PNAS.
- **use_cpu**: if _true_, only CPU will be used, even if the device has usable GPUs.


## Additional scripts and utils
### Time prediction testing script
An additional script is also provided to analyze the results of multiple predictors on time target, on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_time_test*) inside the log folder given as argument.
Script can be launched with the following command:
```
python scripts/predictors_time_testing.py -p {absolute_path_to_logs}/{target_folder(date)}
```

### Controller testing script
Another additional script is provided to analyze the results of multiple controller configurations on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_acc_test*) inside the log folder given as argument.
Since the script use relational imports, you must run this script from the main project folder (outside src), with the -m flag:
```
python scripts/predictors_acc_testing.py -p {absolute_path_to_logs}/{target_folder(date)}
```

### Plot slideshow script
The plot_slideshow.py script is provided to facilitate visualizing related plots in an easier and faster way. To use it, only the log folder must be provided.
An example of the command usage (from src folder):
```
python ./scripts/plot_slideshow.py -p {absolute_path_to_logs}/{target_folder(date)}
```
Close a plot overview to visualize the next one, the program terminates after showing all plots.

If **--save** flag is specified, it will instead save all slides into 'plot_slides' folder, inside the log folder provided as -p argument.

If regressor_testing and/or controller_testing scripts have been run on data contained in selected log folder, also their output plots will be visualized
in additional slides at the end.


### Tensorboard
Trained CNNs have a callback for saving info to tensorboard log files. To access all the runs, run the command:
```
tensorboard --logdir {absolute_path_to_POPNAS_src}/logs/{date}/tensorboard_cnn --port 6096
```
In each tensorboard folder it's also present the model summary as txt file, to have a quick and simple overview of its structure.


## Changelog from original version

### Generated CNN structure changes
- Fix cell structure to be an actual DAG, before only flat cells were generated (it was not possible to use other blocks output
  as input of another block).
- Fix blocks not having addition of the two operations output.
- Fix skip connections (input index -2) not working as expected.
- Tweak child CNN training hyperparameters, to make them more similar to PNAS paper.
- Add ScheduledDropPath in local version (see FractalNet), since it was used in both PNAS and NASNet.
- Allow the usage of flexible kernel sizes for each operation supported by the algorithm.
- Add support for TransposeConvolution operation.


### Equivalent networks detection
- Equivalent blocks are now excluded from the search space, like in PNAS.
- Equivalent models (cell with equivalent structure) are now pruned from search, improving pareto front quality.
  Equivalent models could be present multiple times in pareto front before this change, this should improve a bit the diversity of the models trained.


### Predictors changes
- CatBoost can now be used as time regressor, instead or together aMLLibrary supported regressors.
- Add back the input columns features to regressor, as now inputs really are from different cells/blocks, unlike original implementation.
  Therefore, input values have a great influence on actual training time and must be used by regressor for accurate estimations.
- Use a newer version of aMLLibrary, with support for hyperopt. POPNAS algorithm and the additional scripts also use aMLLibrary multithreading to
  train the models faster, with a default number of threads equal to accessible CPU cores (affinity mask).
- Add another optimizer to LSTM controller, to use two different learning rates (one for B=1, the other for any other B value)
  like specified in PNAS paper.
- Add predictors testing scripts, useful to tune their hyperparameters on data of an already completed run (both time and accuracy).
  These scripts are useful to tune the predictors in case their results are not optimal on given dataset.
- Keras predictors supports hyperparameters tuning via Keras Tuner library. The search space for the parameters is
  defined in each class and can be overridden by each model to adapt it for each specific model.

### Exploration step
- Add an exploration step to POPNAS algorithm. Some inputs and operators could not appear in pareto front
  networks (or appear very rarely) due to their early performance, making the predictors penalizing them heavily also
  in future steps. Since these inputs and operators could be actually quite effective in later cell expansions,
  the exploration step now trains a small set of networks that contains these underused values. It also helps to
  discover faster the value of input values >= 0, since they are unknown in B=1 step and progressively added in future steps.


### Data extrapolation and data analysis
- Add plotter module, to analyze csv data saved and automatically producing relevant plots and metrics while running the algorithm.
  Add also the plot slideshow script to visualize all produced plots easily in aggregated views.
- Add new avg_training_time.csv to automatically extrapolate the average CNN training time for each considered block size.


### Software improvements and refactors
- Migrate code to Tensorflow 2.
- Now use json configuration files, with some optional command line arguments. This approach is much more flexible and makes easier to parametrize
  all the various components of the algorithm run. Many parameters and hyperparameters that were hardcoded in POPNAS initial version are now
  tunable from the json config.
- Implement an actually working run restoring functionality. This allows to resume a previously interrupted run.
- CNN training has been refactored to use Keras model.fit method, instead of using a custom tape gradient method.
  New training method supports ImageGenerators and allows using weight regularization if --wr parameter is provided.
- LSTM controller has been refactored to use Keras API, instead of using a custom tape gradient method.
  This make the whole procedure easier to interpret and also more flexible to further changes and additions.
- Add predictors hierarchy (see _predictors_ folder). Predictor abstract class provides a standardized interface for all regressor methods
  tested during the work. Predictors can be either based on ML or NN techniques, they just need to satisfy the interface to be used during POPNAS
  algorithm and the additional scripts.
- Encoder has been totally refactored since it was a total mess, causing also a lot of confusion inside the other modules.
  Now the state space stores each cell specification as a list of tuples, where the tuples are the blocks (input1, op1, input2, op2).
  The encoder class instead provides methods to encode/decode the inputs and operators values, with the possibility of adding multiple encoders
  at runtime and using them easily when needed. The default encoders are now 1-indexed categorical integers, instead of the 0-indexed used before. 
- Improve immensely virtual environment creation, by using Poetry tool to easily install all dependencies.
- Improve logging (see log_service.py), using standard python log to print on both console and file. Before, text logs were printed only on console.
- Implement saving of best model, so that can be easily trained after POPNAS run for further experiments. A script is provided
  to train the best model.
- Format code with pep8 and flake, to follow standard python formatting conventions.
- General code fixes and improvements, especially improve readability of various code parts for better future maintainability.
  Many blob functions have been finely subdivided in multiple sub-functions and are now properly commented.
  Right now almost the total codebase of original POPNAS version have been refactored, either due to structural or quality changes.


### Command line arguments changes
- Add --cpu option to easily choose between running on cpu or gpu.
- Add --pnas option to run without regressor, making the procedure similar to original PNAS algorithm.
- Add a lot of new configuration parameters, which can be set in the new json configuration file.


### Other bug fixes
- Fix regressor features bug: dynamic reindexing was nullified by an int cast.
- Fix training batch processing not working as expected, last batch of training of each epoch could have contained duplicate images due to how repeat
  was wrongly used before batching.
- Fix tqdm bars for model predictions procedure, to visualize better its progress.



## TODO
- Improve and tweak best model training script.
- Improve the restoring function and investigate potential bugs (especially in prediction and expansion phase it could not work properly, since
  I only wrote the logic to stop it during CNN training, which should be the 90% of the cases).
- Improve quality of plots generated, adding new relevant metrics if useful.
- Generalize on other datasets, right now some logic is basically hardcoded for CIFAR-10 usage, but it shouldn't require much work
  to support other datasets.
