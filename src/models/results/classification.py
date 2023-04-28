from search_space_units import CellSpecification
from .base import BaseTrainingResults, TargetMetric, extract_metric_from_train_histories


class ClassificationTrainingResults(BaseTrainingResults):
    def __init__(self, cell_spec: CellSpecification, training_time: float, inference_time: float, params: int, flops: int,
                 accuracy: float, f1_score: float, top_k_categorical_accuracy: float) -> None:
        super().__init__(cell_spec, training_time, inference_time, params, flops)

        self.accuracy = accuracy
        self.f1_score = f1_score
        self.top_k_categorical_accuracy = top_k_categorical_accuracy

    @staticmethod
    def from_training_histories(cell_spec: CellSpecification, training_time: float, inference_time: float, params: int, flops: int,
                                histories: 'list[dict[str, list]]', multi_output: bool) -> BaseTrainingResults:
        accuracy = extract_metric_from_train_histories(histories, 'accuracy', max, multi_output)
        f1_score = extract_metric_from_train_histories(histories, 'f1_score', max, multi_output)
        top_k_categorical_accuracy = extract_metric_from_train_histories(histories, 'top_k_categorical_accuracy', max, multi_output)

        return ClassificationTrainingResults(cell_spec, training_time, inference_time, params, flops, accuracy, f1_score, top_k_categorical_accuracy)

    @staticmethod
    def from_csv_row(row: list) -> BaseTrainingResults:
        return ClassificationTrainingResults(row[8], *row[3:7], *row[0:3])

    @staticmethod
    def keras_metrics_considered() -> 'list[TargetMetric]':
        return [
            TargetMetric('accuracy', max, results_csv_column='best val accuracy', pareto_predict_csv_column='val score', need_predictor=True),
            TargetMetric('f1_score', max, results_csv_column='val F1 score', pareto_predict_csv_column='val score', need_predictor=True),
            TargetMetric('top_k_categorical_accuracy', max, results_csv_column='val top k accuracy',
                         pareto_predict_csv_column='val score', need_predictor=True)
        ]

    def to_csv_list(self) -> list:
        return [self.accuracy, self.f1_score, self.top_k_categorical_accuracy] + super().to_csv_list()

    def log_results(self):
        self._logger.info('Best accuracy reached: %0.4f', self.accuracy)
        self._logger.info('Best F1 score reached: %0.4f', self.f1_score)
        self._logger.info('Best top K accuracy reached: %0.4f', self.top_k_categorical_accuracy)
        super().log_results()
