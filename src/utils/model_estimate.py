class ModelEstimate:
    '''
    Helper class, basically a struct with a function to convert into array for csv saving
    '''

    def __init__(self, cell_spec: list, score: float, time: float = None, params: int = None):
        self.cell_spec = cell_spec
        self.score = score
        self.time = time
        self.params = params

    def to_csv_array(self):
        cell_structure = f"[{';'.join(map(lambda el: str(el), self.cell_spec))}]"
        return [self.time, self.score, self.params, cell_structure]

    def is_dominated_by(self, other: 'ModelEstimate'):
        ''' Check if this point is dominated by another one provided as argument (regarding Pareto optimality). '''
        return self.score <= other.score and self.time >= other.time and self.params >= other.params

    @staticmethod
    def get_csv_headers():
        return ['time', 'val accuracy', 'params', 'cell structure']
