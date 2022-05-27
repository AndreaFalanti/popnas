class ModelEstimate:
    '''
    Helper class, basically a struct with a function to convert into array for csv saving
    '''

    def __init__(self, cell_spec, score, time):
        self.cell_spec = cell_spec
        self.score = score
        self.time = time

    def to_csv_array(self):
        cell_structure = f"[{';'.join(map(lambda el: str(el), self.cell_spec))}]"
        return [self.time, self.score, cell_structure]

    @staticmethod
    def get_csv_headers():
        return ['time', 'val accuracy', 'cell structure']
