# POPNASv2
Fix and refactor of POPNAS, a neural architecture search method developed for a master thesis by Matteo Vantadori (Politecnico di Milano, accademic year 2018-2019).

## Installation
Virtual environment and dependecies are managed by *poetry*, check out its [repository](https://github.com/python-poetry/poetry) for installing it on your machine.

After installing poetry, optionally run on your terminal:
```
poetry config virtualenvs.in-project true
```
to change the virtual environment generation position directly in the project folder.

To generate the environment, open your terminal inside the repository folder and execute:
```
poetry install
```

After that you should be able to run POPNASv2 with this command:
```
python run.py -b 5 -k 256
```

## Build Docker container
In *docker* folder it's provided a dockerfile to extend an official Tensorflow container with missing pip packages and mount POPNAS source code.
To build the image open the terminal into the *src* folder and execute this command:
```
docker build -f ../docker/Dockerfile -t falanti/popnas:py3.6.9-tf2.6.0gpu .
```

POPNASv2 can then be launched with command (set arguments as you like):
```
docker run falanti/popnas:py3.6.9-tf2.6.0gpu python run.py -b 5 -k 2 -e 1
```

## Command line parameters
TODO

## Changelog from original version
TODO
