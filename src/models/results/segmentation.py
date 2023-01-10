from .base import BaseTrainingResults, TargetMetric, extract_metric_from_train_histories


class SegmentationTrainingResults(BaseTrainingResults):
    def __init__(self, cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
                 accuracy: float, mean_io_u: float) -> None:
        super().__init__(cell_spec, training_time, inference_time, params, flops)

        self.accuracy = accuracy
        self.mean_io_u = mean_io_u

    @staticmethod
    def from_training_histories(cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
                                histories: 'list[dict[str, list]]', multi_output: bool) -> BaseTrainingResults:
        accuracy = extract_metric_from_train_histories(histories, 'accuracy', max, multi_output)
        mean_io_u = extract_metric_from_train_histories(histories, 'mean_io_u', max, multi_output)

        return SegmentationTrainingResults(cell_spec, training_time, inference_time, params, flops, accuracy, mean_io_u)

    @staticmethod
    def from_csv_row(row: list) -> BaseTrainingResults:
        return SegmentationTrainingResults(row[7], *row[2:6], *row[0:2])

    @staticmethod
    def keras_metrics_considered() -> 'list[TargetMetric]':
        return [
            TargetMetric('accuracy', max, results_csv_column='best val accuracy', prediction_csv_column='val score'),
            TargetMetric('mean_io_u', max, results_csv_column='val mean IoU', prediction_csv_column='val score')
        ]

    def to_csv_list(self) -> list:
        return [self.accuracy, self.mean_io_u] + super().to_csv_list()

    def log_results(self):
        self._logger.info("Best accuracy reached: %0.4f", self.accuracy)
        self._logger.info("Best mean IoU reached: %0.4f", self.mean_io_u)
        super().log_results()
