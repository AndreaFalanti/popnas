# POPNASv2
Second version of Pareto-Optimal Progressive Neural Architecture Search (POPNAS) algorithm, which expands
[PNAS](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)
workflow to address a multi-objective optimization problem.
This new version further improves the time efficiency of the search algorithm compared to PNAS and the first version,
while preserving the architecture quality in terms of accuracy.

## Installation
This section provides information for installing all needed software and packages for properly run POPNASv2 on your system.
The easiest way is to use a Docker container, but you can also set up a local environment with _Poetry_ tool.

### Docker container
In _docker_ folder it's provided a dockerfile to extend an official Tensorflow container with project dependencies
and mount POPNAS source code.

To build the image, open the terminal into the _src_ folder and execute this command:
```
docker build -f ../docker/Dockerfile -t falanti/popnas:tf2.7.0gpu .
```

POPNASv2 can then be launched with command (set arguments as you like):
```
docker run -it --rm --name popnas falanti/popnas:tf2.7.0gpu python run.py
```

### Local installation
Make sure your active python version is either 3.7 or 3.8. If you have installed [poetry](https://github.com/python-poetry/poetry),
you can set up the environment by running this command in the project folder:
```
poetry install
```

You can activate the new environment with command:
```
poetry shell
```

Make also sure that *graphviz* is installed in your machine, since it is required to generate plots of Keras models.
Follow the installation instructions at: https://graphviz.gitlab.io/download/.

After that you should be able to run POPNASv2 with this command (in _src_ folder):
```
python run.py
```

To use a GPU locally, you must satisfy Tensorflow GPU hardware and software requirements.
Follow https://www.tensorflow.org/install/gpu instructions to setup your device, make sure
to install the correct versions of CUDA and CUDNN for Tensorflow 2.7 (see https://www.tensorflow.org/install/source#linux).

## Run configuration
### Command line arguments
All command line arguments are optional.
- **-j**: specifies the path of the json configuration to use. If non provided, _configs/run.json_ will be used.
- **-r**: used to restore a previous interrupted run. Specifies the path of the log folder of the run to resume.
- **--name**: used to set a custom folder name for log outputs.

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
- **use_adamW**: use adamW instead of standard L2 regularization
- **drop_path_prob**: defines the max probability of dropping a path in _scheduled drop path_. If set to 0,
  then _scheduled drop path_ is not used.
- **cosine_decay_restart**: dictionary for hyperparameters about cosine decay restart schedule.
  - **enabled**: use cosine decay restart or not (plain learning rate)
  - **period_in_epochs**: first decay period in epochs
  - **[t_mul, m_mul, alpha]**: see [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts)
- **softmax_dropout**: probability of dropping a value in output softmax. If set to 0, then dropout is not used.
  Note that dropout is used on each output when _multi_output_ flag is set.

**CNN architecture parameters**:
- **motifs**: motifs to stack in each CNN. In NAS literature, a motif usually refers to a single cell, here instead it is used to indicate
  a stack of N normal cells followed by a single reduction cell.
- **normal_cells_per_motif**: normal cells to stack in each motif.
- **concat_only_unused_blocks**: when _true_, only blocks' output not used internally by the cell will be used in final concatenation cell output,
  following PNAS and NASNet. If set to _false_, all blocks' output will be concatenated in final cell output.
- **multi_output**: if true, each CNN generated will have an output (GAP + Softmax) at the end of each cell.

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
- **predictions_batch_size**: defines the batch size used when performing both time and accuracy predictions in controller
  update step (predictions about cell expansions for blocks b+1). Incrementing it should decrease the prediction time
  linearly, up to a certain point, defined by hardware resources used.
- **pnas_mode**: if _true_, the algorithm will not use the temporal regressor and pareto front search, making the run very similar to PNAS.
- **use_cpu**: if _true_, only CPU will be used, even if the device has usable GPUs.


## Tensorboard
Trained CNNs have a callback for saving info to tensorboard log files. You can access the data of all the networks
trained by running the command:
```
tensorboard --logdir {absolute_path_to_POPNAS_src}/logs/{folder_name}/tensorboard_cnn
```
In each tensorboard folder it's also present the model summary as txt file and the model graph as a pdf file.
Note that graphs of large networks are very difficult to read, analyzing the summary is advised in these cases.
