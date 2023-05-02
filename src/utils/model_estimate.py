from search_space_units import CellSpecification


class ModelEstimate:
    ''' Helper class, basically a struct with a function to convert into an array for csv saving. '''

    def __init__(self, cell_spec: CellSpecification, score: float, time: float = 0, params: int = 0, inference_time: float = 0):
        self.cell_spec = cell_spec
        self.score = score
        self.time = time
        self.params = params
        self.inference_time = inference_time

    def to_csv_array(self):
        return [self.time, self.score, self.params, self.inference_time, str(self.cell_spec)]

    def is_dominated_by(self, other: 'ModelEstimate'):
        ''' Check if this point is dominated by another one provided as argument (regarding Pareto optimality). '''
        return self.score <= other.score and self.time >= other.time \
            and self.params >= other.params and self.inference_time >= other.inference_time

    @staticmethod
    def get_csv_headers():
        return ['time', 'val score', 'params', 'inference_time', 'cell structure']
