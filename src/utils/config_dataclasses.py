from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CosineDecayRestartDict:
    enabled: bool
    period_in_epochs: int
    t_mul: float
    m_mul: float
    alpha: float


@dataclass
class ResizeDict:
    enabled: bool
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
    operators: 'list[str]'


@dataclass
class SearchStrategyConfig:
    max_children: int
    max_exploration_children: int
    score_metric: str
    additional_pareto_objectives: 'list[str]'


@dataclass
class CnnHpConfig:
    epochs: int
    learning_rate: float
    filters: int
    weight_reg: float
    use_adamW: bool
    drop_path_prob: float
    cosine_decay_restart: CosineDecayRestartDict
    softmax_dropout: float


@dataclass
class ArchitectureParametersConfig:
    motifs: int
    normal_cells_per_motif: int
    block_join_operator: str
    lookback_reshape: bool
    concat_only_unused_blocks: bool
    residual_cells: bool
    multi_output: bool
    se_cell_output: bool = field(default=False)


@dataclass
class DatasetConfig:
    type: str
    name: str
    path: Optional[str]
    classes_count: int
    batch_size: int
    inference_batch_size: int
    validation_size: Optional[float]
    cache: bool
    folds: int
    samples: Optional[int]
    balance_class_losses: bool
    rescale: Optional[bool]
    normalize: Optional[bool]
    resize: Optional[ResizeDict]
    data_augmentation: DataAugmentationDict


@dataclass
class OthersConfig:
    accuracy_predictor_ensemble_units: int
    predictions_batch_size: int
    save_children_weights: bool
    save_children_as_onnx: bool
    train_strategy: str
    pnas_mode: bool = False
    enable_XLA_compilation: bool = False


@dataclass
class RunConfig:
    search_space: SearchSpaceConfig
    search_strategy: SearchStrategyConfig
    cnn_hp: CnnHpConfig
    architecture_parameters: ArchitectureParametersConfig
    dataset: DatasetConfig
    others: OthersConfig
