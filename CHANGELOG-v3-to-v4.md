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

### v3.4.0

Refactor how NetworkGraph is built, coupling it with ModelGenerators.
Other changes include the mapping of the configuration to dataclasses, and the implementation of cached properties on CellSpecification.

- Refactor NetworkGraph to be integrated easily in the ModelGenerator.
  ModelGenerator interface now exposes the "build_model_graph" function, which can be used to create the DAG representation of the entire network.
  In this way, each model generator can create a graph representing their respective model without instantiating it in TF and Keras,
  making it possible to extract quickly the number of parameters.
  The graph could be exploited for additional analysis, e.g., for studying the network partitioning on multiple devices.
  Old modules used for building graphs have been deleted and replaced with this new implementation.
- The configuration JSON dictionary is now mapped to nested dataclasses, making it flexible to changes and error-proof
  (just refactor the dataclass fields, instead of replacing strings and default values in code!).
  Also, type checking by default without extra type annotations :)
- Add cached properties to CellSpecification, making it very time efficient for ModelGenerators and Predictors to retrieve
  the properties needed for their goals (the memory overhead is instead negligible).
  This change also avoids some code duplication.
- Update ModelGenerator to automatically upsample with transpose convolutions the lookback tensors with a lower spatial resolution
  than the target cell shape. This change simplifies the segmentation models and possible future models using encoder-decoder structures.
- Add dependencies for testing, and some unit tests about changed modules.
  In the future, investing some time in tests definition is a much-needed obligation for better stability.
- Update log_service to use a temp directory when a path is not set. Useful for testing and external scripts.
- Update ONNX compilation opset to 13, to make POPNAS consistent with other AI-SPRINT tools.
- Fix bug in time series classification Keras preprocessing model (related to TF update).
- General bug fixes and code cleaning.

### v3.5.0

Add new ModelGenerator for semantic segmentation tasks, based on DeepLab architectures.
Refactor the configuration format for clarity and adds a new optimizer section, which can be used to define the scheduler
and optimizer to use for the neural networks training directly from the config.

- Add new custom layer implementing the Atrous Spatial Pyramid Pooling (ASPP), used in deeplab architectures.
- Add SegmentationFixedDecoderModelGenerator, an alternative model generator for semantic segmentation tasks, which
  macro-architecture is inspired by DeepLab. The encoder is built in a similar way to POPNAS classification networks, and
  terminates with ASPP. The decoder is fixed and uses a lightweight structure similar to DeepLabV3+.
  Furthermore, this model generator works with datasets using sparse labels.
- Update TensorFlow to 2.10.1, to support easier class masking in semantic segmentation tasks. Now a class (void label)
  can be ignored in the loss and mean IoU metric.
  NOTE: TF 2.10.1 has a vectorization issue with Keras augmentation layers, so classification experiments can have some overhead due to this bug.
- Fix major bug in the model selection step, causing the algorithm to not compute correctly the number of parameters
  required by the macro-config alternatives.
- Add weight sampling in the segmentation task, used to balance the loss based on the distribution of the class labels.
  It is activated if the "balance_class_weights" config flag is set to True.
- Add support for instantiating and combining schedulers and optimizers at runtime, based on some structured strings
  provided in the configuration JSON. The strings define the scheduler/optimizer type and eventual optional parameters,
  recognized through regexes. The new logic is more powerful and supports 5 optimizers, 2 schedulers and the optional
  usage of the lookahead mechanism.
- Heavily refactor the JSON configuration format, improving clarity and making some fields optional to avoid clutter.
- Fix sampled models ONNX save using the model without the trained weights. Add also the TF format for convenience.

### v3.6.0
Add new functionalities and configuration options. Refactor how TFDS datasets are loaded and extend support to segmentation tasks.

- Extend class weights for loss balancing for supporting sparse labels. The code automatically detects if one-hot or sparse labels
  are used and compute the weights in a transparent way to the user.
- Add class relabeling, a functionality that allows the substitution of any label with another value.
  A possible use case for this feature is to aggregate multiple classes under a single label.
- Add support for mixed precision (FP16 compute type, FP32 store type). See TF mixed precision guide for further info.
- Make possible to change the default activation function directly from the JSON config. By default, Swish is used, as done in previous tests.
  The config parameter accepts any string identifier accepted by Keras itself.
- Refactor the operator allocators, simplifying the application of weight regularization and the new activation function setting.
- Move onnxruntime to dev dependencies, since not used in actual POPNAS code.
- Reduce the default number of filters used in each branch of the ASPP, set to half the original filters.
  The ratio of filters can be set through a new parameter.
- Change how the architecture derived from the empty cell of DeepLab-like models is built, following the macro-architecture
  of other networks using non-empty cells.
- Fix ImagePreprocessing padding being applied when not necessary.

### v3.7.0
Improve the predictors and fix performance problems in image classification data augmentation due to Keras issues.

- Fix SpektralPredictors not being able to work on batches during predict.
- Adapt KerasPredictors to use learning rate and weight decay schedulers.
- Change KerasPredictor default optimizer to AdamW with learning rate and weight decay scheduled though cosine decay.
  Weight regularization is set to 0 by default, since it has been "replaced" by weight decay which seems to perform similar or better.
- Tune the default hyperparameters of multiple predictors and extend the default epochs of all KerasPredictors.
- Fix predictors dtype to FP64 in all layers, since they seem to benefit from the finer precision.
  Furthermore, mixed precision caused instability issues in predictors, so fixing the precision solves also this issue.
- Add EnsemblePredictor, which acts as a wrapper of multiple predictors to build a heterogeneous ensemble of multiple models.
- Add experimental GIN-LSTM Predictor and experimental losses aimed to improve ranking among predictions.
  Experimental results do not indicate improvements, so they are not used right now.
- Change POPNAS default predictors: the accuracy predictor is now implemented with GIN, while the time predictor with a SVR.
- Change how the data augmentation is performed in image classification tasks.
  The Keras model for data augmentation has been replaced with TF functions, making an upscale + random crop to replace the random translation.
  This is not equivalent to translation, but some paper already used this technique and TF functions are vectorized correctly
  (contrary to the Keras ones), which makes them much faster to execute.
- Fix ASPP layer not using the activation function set in config file (was fixed to Swish).
- Fix the script for running NATS-Bench.
- Change the base docker image from official TF image to the one provided by NVIDIA (same TF version).
  There was a "bus error" issue when using the previous image in Docker in multi-GPU settings, the new image instead seems to work fine.

### 3.7.1
Small refactors on json config default paths, rename aMLLibrary config file and fix its relative path in predictor initializer.

### 3.7.4
Integrate Neptune into POPNAS, set label smoothing as a configuration option, and fix multiple bugs.

- Add integration with Neptune. If the related environment variables are provided, POPNAS will log the experiments on Neptune,
  making it possible to access them on cloud and share them among multiple persons.
- Update segmentation dataset preprocessing script to allow resizing images to square crops of fixed dimension.
- Extend DeepLab-like architecture graphs to consider ASPP and decoder, so that the number of parameters can be extracted correctly.
- Label smoothing was previously always applied in post-search procedures. Since this setting works only on one-hot labels,
  it is now applied only in classification tasks if a new parameter is set in the JSON config. This change also enables label
  smoothing during search.
- Fix loss always set to CategoricalCrossentropy in post-search procedures. Now they use the loss function of the model generator.
- Fix DeepLab-like architectures pruning ASPP and decoder when using the "multi-output" setting.
- Fix TFDS train+val and test datasets not working correctly.
- Fix cutout size not being fixed to a multiple of 2, causing an error when odd.

### 3.7.7
Add new scripts to analyze the results of segmentation networks and their inference time. Perform multiple fixes and improvements.

- Add a script for visualizing the predictions made by a segmentation model, as an image comparison of the original image and mask with the prediction.
- Add a script for measuring the inference time of an ONNX model for segmentation tasks, and check the estimated FPS.
- Tweak SVR predictor configuration
- Fix mask padding using class 0 instead of ignored class
- Fix filters argument not being applied correctly in last training script
- Update ONNX opset to 17 and the ONNX dependencies, moving the runtime from dev to main dependencies
- Improve the generation of segmentation datasets and the handling of TFDS datasets
- Fix a potential circular dependency between search space and other modules
- Disable XLA compilation override in default post-search configurations
- Add a new README focused on scripts (see the _scripts_ folder)
- Other minor fixes

### 3.8.0
Update Pareto front to support inference time, add zoomed convolution operator and warmup scheduler wrapper.

- Add a new predictor for computing inference time. This predictor is instantiated only if "inference_time" is provided in the additional
  Pareto metrics of the JSON config. Inference predictor is an SVR such as the training time predictor, using the same feature set but calibrating
  the dynamic reindex map on the inference times of the specular monoblocks.
- Add ZoomedConvolution operator, which can be especially useful in segmentation tasks to optimize the inference time of the network while
  preserving a high receptive field (see FasterSeg paper).
- Add a custom LearningRateScheduler called WarmupSchedulerWrapper, which wraps an already existent Keras scheduler and applies an initial warmup
  for the given number of steps, where the learning rate increases linearly to the target value.
  After the warmup, the wrapped scheduler is used. It is a nice decorator for Keras which is transparent to the user.
- Fix images being upsampled in image classification tasks even when data augmentation was disabled. The images are upsampled just to perform
  random crop to original dimension during augmentation, to simulate zoom and translation of the image, so it is avoided when augmentation
  is not performed.

### 3.8.1
Refine and fix some features added with version 3.8.0.

- Improve inference time measurement, using more batches to reduce noise and discarding the first 3 measurements (instead of only the first one),
  which tends to be noisier. Also, it uses the training set instead of validation during inference, since in some cases validation images could be of
  different sizes, complicating the estimation of the inference time (training samples instead always have the same size for batch purposes).
- Fix bug regarding number of classes when relabelling dictionary was configured in TSC tasks
- Fix serialization of WarmupSchedulerWrapper
- Update hyperparameters of inference time SVR predictor, since the regularization factors did not work well for the new domain.
- Fix ZoomedConvolutions for networks with non-fixed input resolutions

### 3.8.2
Perform small bug fixes, update the README.

This should be my last POPNAS release, since my contract expires today and I will no longer be officially
related to the AI-SPRINT project.
I hope people will enjoy trying out POPNAS and to have made the setup and configuration as intuitive as possible
(even if they are really complex, I know). So long folks, take care of POPNAS for me :)

- Update the inference time script to support all tasks covered by POPNAS.
- Fix ZoomedConvolution serialization.
- Disallow the automatic generation of Pareto front with > 3 dimensions.
- Set labels to tf.uint8 when applying one-hot encoding in the dataset preprocessor.
- Update README to make it more exhaustive and detailed.