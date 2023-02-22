# Changelog from v3.0 to v3.?

In this file I provide a summary of the new features and major refactors performed between the third version (on arXiv as preprint) and
the newest version of POPNAS, currently in development.

### v3.1.0

A huge refactor of the whole repository structure, to make the algorithm more flexible to new use cases and increase code readability and consistency.
- Add ModelGenerator hierarchy, which takes care of instantiating models for different types of tasks. 
  Before this version, only classification tasks were addressed, the idea is now to extend POPNAS to multiple supervised tasks.
- Add TrainingResults hierarchy and TargetMetric class, used to specify which metrics are considered by the model generators and making it possible
  to extract the metrics from the Keras results for logging purposes.
  Metrics were previously hardcoded, and the JSON configuration just specified which subset of metrics to consider in the Pareto front.
  The new implementation allows defining metrics for each problem without tampering the main algorithm logic.
- Adapt POPNAS modules to use the ModelGenerator, instantiating the correct one for the task through a strategy pattern.
- Adapt all modules and post search scripts to use the TargetMetric instances, handling the metrics dynamically in both logs, csv,
  and plots generation.
- Move the custom callbacks to a new subpackage under *models*, and adjust to correctly follow Keras implementation.
- Add DatabaseGenerator and ModelGenerator prototypes for addressing segmentation models, but are still untested and need revision. They shouldn't
  have been added to this release since they are incomplete, but they were a nice way of testing if the new hierarchies are flexible enough :).
- Move most file writing functions to the new file_writer module.
- Set up aMLLibrary as an external dependency managed through git submodules, and update it to the latest public version.

### v3.1.1

Update the Poetry lock file to the latest version (Poetry >= 1.2.0 is required to read it), fix the dependencies and the docker image generation.

### v3.2.0

Performs a major update of the dependencies.
It also introduces a new operator and some other structural refactors.
- Update TensorFlow from v2.7.3 to v2.8.4
- Update supported python versions from (>=3.7.2, <3.9) to (>=3.8, <3.11) 
- Update other dependencies to recent versions
- Add the Squeeze-and-Excitation layer as an operator for POPNAS cells. Previously, it could only be applied to cell outputs.
- Add CellSpecification and BlockSpecification classes, which represent explicitly POPNAS phenotype/genotype.
  These classes are extended with their respective utility functions, making it easier to work with these specific types
  instead of using lists of tuples.
- Fix legacy Keras imports, updating them to the "standard" ones.
- Replace MeanIoU with OneHotMeanIoU in the segmentation models prototype.

### v3.3.0

Finalize the implementation of dataset and model generators for the semantic segmentation task. Include also some minor improvements.
- Introduce the WrappedTensor class, used to keep track of important tensors for POPNAS architectures along with their shape.
  The shape is computed a priori, and often it is expressed as a ratio of the original input dimension, making it possible to verify
  shape incompatibilities even when the input shape is not completely defined
  (some dimensions are None, working as fully convolutional neural networks).
- Update ModelGenerators to use the WrappedTensor class, making it possible to not specify the spatial resolution of the input, and therefore using inputs
  of different sizes. Of course, the macro-architecture must be fully convolutional to exploit this behavior.
- Update SegmentationModelGenerator macro-structure to use upsample cells, incorporating the previously defined upsample units.
  This change makes the structure more compatible with skip lookbacks.
  Also, add support for multi-outputs for segmentation.
- Change the preprocessing script for segmentation datasets, merging the two previous scripts and adding the possibility of resizing
  the images and masks so that the smallest axis has the specified size (keeping the original aspect ratio).
- Improve and fix the ImageSegmentationDatasetGenerator, correctly handling the preprocessing and data augmentation on ragged tensors and applying
  the same transformations to both images and masks, which were bugged in the previous minor versions.
- Move top_k_categorical_accuracy to classification models search metrics, instead of inserting it in post-search procedures.
- Add support for "mean_iou" as score metric.
- Disable graph saving in tensorboard callback, saving a lot of storage space per experiment.
- Make run_e2e execute the various scripts as processes, as done in TSC archives script.

### v3.3.3

Add Flask server to execute and manage POPNAS processes in remote deployments. Include also bug fixes and adjustments.
- The new Flask server is provided in src/server folder. The API is pretty simple, and it is designed to support the main backend in the web deployment.
- Fix wrong rank ordering in Pareto front plots.
- Improve plot slides generation in the plot_slideshow script.
- Fix empty cell always using "accuracy" as the score metric, even if a different one was provided in the configuration.
- Add automatic summary info extraction from both search and post-processing procedures (see experiments_summary module).
- Add a configuration option for XLA compilation (experimental, need more testing).
