# Additional scripts
This folder contains additional scripts that can be used to further analyze and process the neural networks found
during the POPNAS search procedure.
While _model selection_, _last training_, and _evaluate network_, can be integrated into the main search process when launched from
the `run_e2e.py` entrypoint, the other scripts provide additional functionalities to POPNAS users.


### Model selection script
This script can be used to extensively train the best architectures found during a POPNAS search or a custom cell specification.
It uses a configuration file (by default _model_selection_training.json_), which is a subset of the search algorithm JSON file and its properties
override the ones used during search.

If the `-params` option is set, the script also performs a pseudo-grid-search to tune the macro architectures of the selected cells, to generate additional
architectures to train which satisfy the provided parameters range (e.g. "2.5e6;3.5e6", must be semicolon separated).

Models and training procedure are slightly altered compared to search time: each model uses a secondary exit on the cell placed at 2/3 of total
model length.
If using the default config, scheduled drop path and cutout are also applied to improve the generalization during the extended number of epochs.

The script can be launched with the following command:
```
python scripts/model_selection_training.py -p {path_to_log_folder} -j {path_to_config_file}
```

### Last training script
This script is designed for training the best architecture found during the model selection, posterior to the NAS run.
It uses a configuration file (by default _last_training.json_), which is a subset of the search algorithm JSON file and its properties
override the ones used during search.

The model and training procedure are tweaked as done during model selection, but in this case the model is training on both
the training and validation data, without reserving a validation set.
It is important to check that the model generalizes well during the model selection phase, to not incur in overfitting.

Here, multiple command line arguments must be provided, to build the exact model without altering the configuration file:
- `-b`: the desired batch size
- `-f`: the model starting filters
- `-m`: the number of motifs used in the model
- `-n`: the number of normal cells to stack per motif
- `-spec`: the cell specification, in format "[(block0);(block1);...]"

The script can be launched with the following command (change parameters accordingly):
```
python scripts/last_training.py -p {path_to_log_folder} -j {path_to_config_file} -b 256 -f 24 -m 3 -n 2 -spec "[(-2, '1x3-3x1 conv', -1, '1x5-5x1 conv');(-1, '1x3-3x1 conv', 0, '2x2 maxpool')]"
```

### Evaluation script
Using this script, it is possible to evaluate a previously trained model on the test set of the dataset selected in the configuration.
The information is saved in the _eval.txt_ file inside the model folder, together with the confusion matrix.

The script can be launched with:
```
python scripts/evaluate_network.py -p {path_to_log_folder} -f {model_folder}
```


### Time prediction testing script
An additional script is provided to analyze the results of multiple time predictors, on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_time_test*) inside the log folder given as argument.

The script can be launched with the following command:
```
python scripts/predictors_time_testing.py -p {path_to_log_folder}
```

### Accuracy predictor testing script
Another additional script is provided to analyze the results of multiple accuracy predictor configurations, on the data gathered in a POPNAS run.
The script creates an additional folder (*pred_acc_test*) inside the log folder given as argument.

The script can be launched with the following command:
```
python scripts/predictors_acc_testing.py -p {path_to_log_folder}
```

### Plot slideshow script
This script is provided to facilitate the visualization of related plots, aggregating them in macro plots.
To use it, only the log folder must be provided.

An example of the command usage (from src folder):
```
python scripts/plot_slideshow.py -p {path_to_log_folder}
```
Close a plot overview to visualize the next one, the program terminates after showing all plots.

If the `--save` flag is specified, it will instead save all slides into _plot_slides_ folder, inside the log folder provided as -p argument.

If any of the _predictor testing_ scripts have been run on data contained in selected log folder,
their plots will be appended in additional slides.


### Save cell image script
A simple script which can save the graphical representation of the given cell specification in PDF and DOT formats.

The script can be executed with the following command:
```
python scripts/save_cell_graph_image.py -p {path_to_save_folder} -spec {cell specification as str, e.g., "[(-2, 2x2 maxpool, -2, 3x3 conv);(0, 3x3 conv, 0, 5x5 conv);(0, 8r SE, 1, 8r SE)]"}
```


### Segmentation results script
This script outputs the predictions of a segmentation network, applying the mask over the original image and comparing it with the
true labels.
It is a nice way to visualize the results and check for potential inaccuracies in both masks and network predictions.

The script can be executed with the following command:
```
python scripts/check_segmentation_inference_results.py -p {path to model output of last training script}
```


### POPNAS generator for NPZ segmentation dataset script
Script used to generate a standard format readable by POPNAS to address semantic segmentation datasets.
It expects separate folders containing the sample images and the masks, with the same filename,
and converts them to npz arrays for each dataset split.

The script can be executed with the following command:
```
python scripts/create_popnas_segmentation_dataset.py -p {folder path of each split to process} -m {mode for converting masks, e.g. "R:R" if masks are RGB and you want to keep them as a sparse label}
```


### Segmentation inference time script
This script checks out the inference speed of the target ONNX model on the device hardware.
It is meant mainly for real-time segmentation tasks, where a certain number of frame-per-second (FPS) is required for the
network to be deployed.

The script can be executed with the following command:
```
python scripts/compute_segmentation_inference_time.py -p {path to model output of last training script} -d {path to dataset} --onnx_only
```
