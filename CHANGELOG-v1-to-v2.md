# Changelog from v1.0 to v2.0

The differences between the two major versions are shown here in details at a macro level.
Most changes and WIP of versions 2.0+ are not tracked in this changelog (versions 2.0+ are refinement posterior to WCCI publication).

The software structure has been completely refactored and the results have been improved in every aspect compared to the first version, making
POPNASv1 obsolete and not worthy to use anymore. Still, the changes are documented for completeness.

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
- Equivalent models (cell with equivalent structure) are now pruned from search, improving Pareto front quality.
  Equivalent models could be present multiple times in Pareto front before this change, this should improve a bit the diversity of the models trained.


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
- Add an exploration step to POPNAS algorithm. Some inputs and operators could not appear in Pareto front
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


### Other bug fixes
- Fix regressor features bug: dynamic reindexing was inaccurate due to an int cast that was totally unnecessary since the dynamic reindex is
  designed to be a float value.
- Fix training batch processing not working as expected, last batch of training of each epoch could have contained duplicate images
  due to how repeat was wrongly used before batching.
- Fix tqdm bars for model predictions procedure, to visualize better its progress.