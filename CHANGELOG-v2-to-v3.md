# Changelog from v2.0 to v3.0

In this file I provide a summary of the new features and major refactors performed between the IEEE IJCNN version (v2) and
the third version of POPNAS, currently available as preprint on arXiv:
- extension to time series classification
- TPU support
- post-search training script improvements
    -  model selection phase, with macro architecture tuning
    -  last training phase, with whole dataset (train+val)
    -  model and hyperparameter tweaks, like the secondary exit and label smoothing in loss
- new search architecture parameters and improvements
    - block join operator
    - lookback reshape
    - multi-output
    - residual cells
- possibility to specify F1 score as target metric score, instead of accuracy
- weighted loss for unbalanced datasets
- scheduled drop path improvements
- extension to 3 Pareto objectives to support "params" objective
- POPNASv2 adaptation for NAS-Bench-201 (NATS-Bench TSS)
- dataset pipeline improvements
- add prototype for CvT (transformer-based operator)
- add support for new predictor implementations, like LGBM and graph neural networks (experimental)
- add entrypoint for running end-to-end experiments, running the NAS procedure, model selection and final training in sequence
- add entrypoint for running experiments on a vast selection of time series classification datasets
- small tweaks on time series classification default configuration
- separate operators in layers and allocators hierarchy, to increase the flexibility of their allocation when the tensors require
  to be reshaped by the operator (e.g. in the reduction cells mainly, some operators could diverge from normal implementation).
  This makes implementing operators more straightforward in the future.
- add LSTM, GRU and dilated convolutions to the supported operators
- add scripts for retrieving data from the experiments and saving them into csv files
- fix logging not instantiating correctly additional custom loggers
- add multi-GPU execution and tweak dataset format for TPU
- ...and probably other minor things which I forgot to mention 😅

