# POPNAS
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

After that you should be able to run POPNAS with this command:
```
python run.py -b 5 -k 256
```

## Command line parameters
TODO
