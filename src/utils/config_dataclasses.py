from dataclasses import dataclass, field
from typing import Optional, List

# NOTE: to keep support for python 3.8, do not use types with '', the dacite package will incorrectly consider them as forward references.
#  Use instead the classes from typing package.


@dataclass
class LookaheadDict:
    sync_period: int
    slow_step_size: float


@dataclass
class OptimizerDict:
    type: str
    scheduler: Optional[str]
    lookahead: Optional[LookaheadDict]


@dataclass
class ResizeDict:
    width: int
    height: int


@dataclass
class DataAugmentationDict:
    enabled: bool
    perform_on_gpu: bool
    use_cutout: bool = False


@dataclass
class SearchSpaceConfig:
    blocks: int
    lookback_depth: int
    operators: List[str]


@dataclass
class SearchStrategyConfig:
    max_children: int
    max_exploration_children: int
    score_metric: str = 'accuracy'
    additional_pareto_objectives: List[str] = field(default_factory=lambda: ['time'])


@dataclass
class TrainingHyperparametersConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    drop_path: float
    softmax_dropout: float
    optimizer: OptimizerDict


@dataclass
class ArchitectureHyperparametersConfig:
    filters: int
    motifs: int
    normal_cells_per_motif: int
    block_join_operator: str
    lookback_reshape: bool
    concat_only_unused_blocks: bool
    residual_cells: bool
    multi_output: bool
    se_cell_output: bool = False


@dataclass
class DatasetConfig:
    type: str
    name: str
    path: Optional[str]
    classes_count: int
    ignore_class: Optional[int]
    batch_size: int
    inference_batch_size: int
    validation_size: Optional[float]
    cache: bool
    folds: int
    samples: Optional[int]
    rescale: Optional[bool]
    normalize: Optional[bool]
    class_labels_remapping: Optional[dict[str, int]]
    resize: Optional[ResizeDict]
    data_augmentation: DataAugmentationDict
    balance_class_losses: bool = False


@dataclass
class OthersConfig:
    accuracy_predictor_ensemble_units: int
    predictions_batch_size: int
    train_strategy: str
    save_children_weights: bool = False
    save_children_models: bool = False
    pnas_mode: bool = False
    enable_XLA_compilation: bool = False
    use_mixed_precision: bool = False


@dataclass
class RunConfig:
    search_space: SearchSpaceConfig
    search_strategy: SearchStrategyConfig
    training_hyperparameters: TrainingHyperparametersConfig
    architecture_hyperparameters: ArchitectureHyperparametersConfig
    dataset: DatasetConfig
    others: OthersConfig
