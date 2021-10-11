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

## GPU support
To use GPU locally, you must satisfy Tensorflow GPU hardware and software requirements.
Follow https://www.tensorflow.org/install/gpu instructions to setup your device, make sure
to install the correct versions of CUDA and CUDNN for Tensorflow 2.5 (see https://www.tensorflow.org/install/source#linux).

## Build Docker container
In *docker* folder it's provided a dockerfile to extend an official Tensorflow container with project required pip packages and mount POPNAS source code.
To build the image, open the terminal into the *src* folder and execute this command:
```
docker build -f ../docker/Dockerfile -t falanti/popnas:py3.6.9-tf2.6.0gpu .
```

POPNASv2 can then be launched with command (set arguments as you like):
```
docker run falanti/popnas:py3.6.9-tf2.6.0gpu python run.py -b 5 -k 2 -e 1 --cpu
```


## Command line arguments
**Required arguments:**
- **-b**: defines the maximum amount of blocks B a cell can contain.
- **-k**: defines the amount of top-K cells the algorithm picks up to expand at the next iteration.

**Optional arguments:**
- **-d**: defines the Python file the program should use as dataset, the default dataset is CIFAR-10.
- **-e**: defines for how many epochs E each child network has to be trained, the default value is 20.
- **-s**: defines the batch size dimension of the dataset, the default value is 128.
- **-l**: defines the learning rate of the child CNN networks, the default value is 0.01 (PNAS).
- **-h**: defines how many times a child network has to be trained from scratch, each time with a different dataset train-validation split,
  in order to minimize the accuracy dependence of the child networks on the splitting.
  The default value is 1, if it is set as higher the resulting accuracy is the arithmetic mean of all the accuracies.
- **-r**: if the user specifies this argument, the algorithm will restore a previous run.
  The correct checkpoint B value, i.e. the number of blocks per cell, needs to be indicated in -c, while the run to recover has to be specified in -t.
- **-c**: defines the checkpoint B value from which restart if the argument -r is specified in the input command.
- **-d**: defines the log folder to restore, if the argument -r is specified in the input command. The string is encoded as *yyyy-MM-dd-hh-mm-ss*.
- **-f**: defines the initial number of filters to use. Defaults to 24.
- **-wn**: defines the L2 regularization factor to use in CNNs. Defaults to None (not applied if not provided).
- **--cpu**: if specified, the algorithm will use only the cpu, even if a gpu is actually available. Must be specified if the host machine has no gpu.
- **--abc**: short for "all blocks concatenation".
  If specified, all blocks' output of a cell will be used in concatenation at the end of a cell to build the cell output,
  instead of concatenating only block outputs not used by other blocks (that is the PNAS implementation behavior, enabled by default).
- **--pnas**: if specified, the algorithm will not use a regressor, disabling time estimation.
  This will make the computation extremely similar to PNAS algorithm.


## Tensorboard
Trained CNNs have a callback for saving info to tensorboard log files. To access all the runs, run the command:
```
tensorboard --logdir {absolute_path_to_POPNAS_src}\logs\{date}\tensorboard_cnn --port 6096
```
In each tensorboard folder it's also present the model summary as txt file, to have a quick and simple overview of its structure.


## Additional scripts
### Regressor testing script
An additional script is also provided to analyze the results of multiple regressors on the data gathered in a POPNAS run.
The script creates an additional folder (*regressors_test*) inside the log folder given as argument.
Since the script use relational imports, you must run this script from the main project folder (outside src), with the -m flag:
```
python -m src.scripts.regressor_testing -p {absolute_path_to_logs}\{target_folder(date)}
```

### Controller testing script
Another additional script is provided to analyze the results of multiple controller configurations on the data gathered in a POPNAS run.
The script creates an additional folder (*controllers_test*) inside the log folder given as argument.
Since the script use relational imports, you must run this script from the main project folder (outside src), with the -m flag:
```
python -m src.scripts.controller_testing -p {absolute_path_to_logs}\{target_folder(date)}
```

### Plot slideshow script
The plot_slideshow.py script is provided to facilitate visualizing related plots in an easier and faster way. To use it, only the log folder must be provided.
An example of the command usage (from src folder):
```
python .\scripts\plot_slideshow.py -p {absolute_path_to_logs}\{target_folder(date)}
```
Close a plot overview to visualize the next one, the program terminates after showing all plots.

If **--save** flag is specified, it will instead save all slides into 'plot_slides' folder, inside the log folder provided as -p argument.

If regressor_testing and/or controller_testing scripts have been run on data contained in selected log folder, also their output plots will be visualized
in additional slides at the end.


## Changelog from original version
- Fix cell structure to be an actual DAG, before only flat cells were generated (it was not possible to use other blocks output as input of another block).
- Fix blocks not having addition of the two operations output.
- Fix skip connections (input index -2) not working as expected.
- Equivalent blocks are now excluded from the search space, like in PNAS.
- Equivalent models (cell with equivalent structure) are now pruned from search, improving pareto front quality.
  Equivalent models could be present multiple times in pareto front before this change, this should improve a bit the diversity of the models trained.
- Implement saving of best model, so that can be easily trained after POPNAS run for further experiments
  A script is provided to train the best model.
- Add the input columns to regressor, as now inputs really are from different cells/blocks, unlike original implementation.
  Therefore, input values have a great influence on actual training time and must be used by regressor for accurate estimations.
- Fix regressor features bug: dynamic reindexing was nullified by an int cast.
- CatBoost can now be used as time regressor, instead of aMLLibrary supported regressors.
- Add plotter module, to analyze csv data saved and automatically producing relevant plots and metrics while running the algorithm.
  Add also the plot slideshow script to visualize all produced plots easily in aggregated views.
- Add regressor and controller testing scripts, useful to tune their hyperparameters on data of an already completed run.
  These scripts are useful to tune the predictors in case their results are not optimal on given dataset.
- Migrate code to Tensorflow 2.
- CNN training has been refactored to use Keras model.fit method, instead of using a custom tape gradient method.
  New training method supports ImageGenerators and allows using weight regularization if --wn parameter is provided.
- LSTM controller has been refactored to use Keras API, instead of using a custom tape gradient method.
  This make the whole procedure easier to interpret and also more flexible to further changes and additions.
- Encoder has been totally changed as it was a total mess, causing also a lot of confusion inside the other modules.
  Now the state space stores each cell specification as a list of tuples, where the tuples are the blocks (input1, op1, input2, op2).
  The encoder class instead provides methods to encode/decode the inputs and operators values, with the possibility of adding multiple encoders
  at runtime and using them easily when needed. The default encoders are now categorical, so 1-indexed integers, instead of the 0-indexed used before.
- Improve immensely virtual environment creation, by using Poetry tool to easily install all dependencies.
- Improve logging (see log_service.py), using standard python log to print on both console and file. Before, text logs were printed only on console.
- Add --cpu option to easily choose between running on cpu or gpu.
- Add --pnas option to run without regressor, making the procedure similar to original PNAS algorithm.
- Add --abc and -f options, to make cell structure more configurable and flexible.
- Tweak both controller and child CNN training hyperparameters, to make them more similar to PNAS paper.
- Print losses on both console and file during CNN and controller training, to make easier the analysis of the training procedure while the algorithm
  is running.
- Fix training batch processing not working as expected, last batch of training of each epoch could have contained duplicate images due to how repeat
  was wrongly used before batching.
- Add another optimizer to LSTM controller, to use two different learning rates (one for B=1, the other for any other B value) like specified in PNAS paper.
- Fix tqdm bars of CNN training and add tqdm bars to LSTM training and model predictions procedure for better progress visualization.
- Add new avg_training_time.csv to automatically extrapolate the average CNN training time for each considered block size.
- Format code with pep8 and flake, to follow standard python conventions.
- General code fixes and improvements, especially improve readability of various code parts for better future maintainability.
  Many blob functions have been finely subdivided in multiple subfunctions and properly commented.


## TODO
- Improve and tweak best model training script.
- Improve quality of plots generated, adding new relevant metrics if useful.
- Investigate the "Model failed to serialize as JSON. Ignoring... " warning triggered by tensorboard call.
  It seems to not alter the program flow, but it's worth a check.
- Logic for handling -h parameter was faulty from the start. Since this functionality has never been used, it's possible to deprecate it or otherwise
  it should be completed.
- Restore procedure logic must be revisited and fixed, it was probably not working properly from the start. This functionality has never been
  used, but it can be nice to have in case the algorithm would take a lot of time on larger dataset.
- Generalize on other datasets, right now some logic is basically hardcoded for CIFAR-10 usage, but it shouldn't be too difficult
  to support other datasets.
