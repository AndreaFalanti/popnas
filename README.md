# POPNASv2
Fix and refactor of POPNAS, a neural architecture search method developed for a master thesis by Matteo Vantadori (Politecnico di Milano, academic year 2018-2019).

## Installation
Virtual environment and dependecies are managed by *poetry*, check out its [repository](https://github.com/python-poetry/poetry) for installing it on your machine.
You need to have installed python version 3.6.9 or 3.7.4 (advised for windows) for building a valid environment (other versions could work, but are not tested).
To install and manage the python versions and work with *poetry* tool, it's advised to use [pyenv](https://github.com/pyenv/pyenv) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) based on your system.

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
- **-h**: defines how many times a child network has to be trained from scratch, each time with a different dataset splitting into train set and validation set, in order to minimize the accuracy dependence of the child networks on the splitting. The default value is 1, if it is set as higher the resulting accuracy is the arithmetic mean of all the accuracies.
- **-r**: if the user specifies this argument, the algorithm will restore a previous run. The correct checkpoint B value, i.e. the number of blocks per cell, needs to be indicated in -c, while the run to recover has to be specified in -t.
- **-c**: defines the checkpoint B value from which restart if the argument -r is specified in the input command.
- **-d**: defines the log folder to restore, if the argument -r is specified in the input command. The string is encoded as *yyyy-MM-dd-hh-mm-ss*.
- **-f**: defines the initial number of filters to use. Defaults to 24.
- **--cpu**: if specified, the algorithm will use only the cpu, even if a gpu is actually available. Must be specified if the host machine has no gpu.
- **--abc**: short for "all blocks concatenation". If specified, all blocks' output of a cell will be used in concatenation at the end of a cell to build the cell output, instead of concatenating only block outputs not used by other blocks (that is the PNAS implementation behavior, enabled by default).

## Changelog from original version
TODO
