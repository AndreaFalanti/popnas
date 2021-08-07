# POPNASv2
Fix and refactor of POPNAS, a neural architecture search method developed for a master thesis by Matteo Vantadori (Politecnico di Milano, accademic year 2018-2019).

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

## Command line parameters
TODO

## Changelog from original version
TODO
