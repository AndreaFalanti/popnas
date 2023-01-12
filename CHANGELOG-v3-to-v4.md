# Changelog from v3.0 to v3.?

In this file I provide a summary of the new features and major refactors performed between the third version (on arXiv as preprint) and
the newest version of POPNAS, currently in development.

### v3.1

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
