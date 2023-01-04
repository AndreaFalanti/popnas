from .base import BaseTrainingResults, MetricTarget, extract_metric_from_train_histories


class SegmentationTrainingResults(BaseTrainingResults):
    def __init__(self, cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
                 accuracy: float, m_iou: float) -> None:
        super().__init__(cell_spec, training_time, inference_time, params, flops)

        self.accuracy = accuracy
        self.m_iou = m_iou

    @staticmethod
    def from_training_histories(cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
                                histories: 'list[dict[str, list]]', multi_output: bool) -> BaseTrainingResults:
        accuracy = extract_metric_from_train_histories(histories, 'accuracy', max, multi_output)
        m_iou = extract_metric_from_train_histories(histories, 'm_iou', max, multi_output)

        return SegmentationTrainingResults(cell_spec, training_time, inference_time, params, flops, accuracy, m_iou)

    @staticmethod
    def from_csv_row(row: list) -> BaseTrainingResults:
        return SegmentationTrainingResults(row[7], *row[2:6], *row[0:2])

    @staticmethod
    def metrics_considered() -> 'list[MetricTarget]':
        return [MetricTarget('accuracy', max), MetricTarget('m_iou', max)]

    @staticmethod
    def get_csv_headers() -> 'list[str]':
        return ['best val accuracy', 'val mean IoU'] + super().get_csv_headers()

    def to_csv_list(self) -> list:
        return [self.accuracy, self.m_iou] + super().to_csv_list()
