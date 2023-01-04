from .base import BaseTrainingResults, MetricTarget, extract_metric_from_train_histories


class ClassificationTrainingResults(BaseTrainingResults):
    def __init__(self, cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
                 accuracy: float, f1_score: float) -> None:
        super().__init__(cell_spec, training_time, inference_time, params, flops)

        self.accuracy = accuracy
        self.f1_score = f1_score

    @staticmethod
    def from_training_histories(cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
                                histories: 'list[dict[str, list]]', multi_output: bool) -> BaseTrainingResults:
        accuracy = extract_metric_from_train_histories(histories, 'accuracy', max, multi_output)
        f1_score = extract_metric_from_train_histories(histories, 'f1_score', max, multi_output)

        return ClassificationTrainingResults(cell_spec, training_time, inference_time, params, flops, accuracy, f1_score)

    @staticmethod
    def from_csv_row(row: list) -> BaseTrainingResults:
        return ClassificationTrainingResults(row[7], *row[2:6], *row[0:2])

    @staticmethod
    def metrics_considered() -> 'list[MetricTarget]':
        return [MetricTarget('accuracy', max), MetricTarget('f1_score', max)]

    @staticmethod
    def get_csv_headers() -> 'list[str]':
        return ['best val accuracy', 'val F1 score'] + BaseTrainingResults.get_csv_headers()

    def to_csv_list(self) -> list:
        return [self.accuracy, self.f1_score] + super().to_csv_list()
