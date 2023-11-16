from NumberType import NumberType, get_number_type_by_name
import numpy as np
from HalfMatrix import HalfMatrix

class Matrix:
    @staticmethod
    def get_number_type():
        return NumberType
    
    def __init__(self, options):
        self.width = options.get('width')
        self.height = options.get('height')
        self.feature_amount = options.get('feature_amount', 1)
        self.number_type = get_number_type_by_name(options.get("number_type", NumberType.FLOAT32))
        self.sample_duration = options.get('sample_duration', 1)
        self.length = self.width * self.height * self.feature_amount

        if 'buffer' in options:
            self.data = np.zeros(options['buffer'], dtype=np.uint8)
            assert self.length == len(self.data)
        else:
            self.data = np.zeros(self.length, dtype=np.uint8)

    @staticmethod
    def from_half_matrix(half_matrix: HalfMatrix):
        matrix = Matrix({
            'width': half_matrix.size,
            'height': half_matrix.size,
            'number_type': half_matrix.number_type,
            'sample_turation': half_matrix.sample_duration,
        })
        matrix.fill(half_matrix.get_value_mirrored)
        return matrix
    
    @staticmethod
    def from_(matrix, options=None):
        if options is None:
            options = {}
        if isinstance(matrix, HalfMatrix):
            return Matrix.from_half_matrix(matrix)
        feature_amount = matrix.feature_amount
        number_type = matrix.number_type
        sample_duration = matrix.number_type

        return HalfMatrix({
            'width': matrix.width,
            'height': matrix.height,
            'feature_amount': feature_amount, 
            'number_type': number_type, 
            'sample_duration': sample_duration
        })

    def fill(self, callback):
        for y in range(self.height):
            for x in range(self.width):
                    self.data[(y * self.width + x) * self.feature_amount] = callback(x, y)
        
    def get_value_normalized(self, x, y, f=0):
        index = (y * self.width + x) * self.feature_amount + f
        return self.data[index] / self.number_type.value['scale']
    
    def get_value(self, x, y, f=0):
        index = (y * self.width + x) * self.feature_amount + f
        return self.data[index]
    
    def get_value_mirrored(self, x, y, f=0):
        return self.get_value(x, y, f)