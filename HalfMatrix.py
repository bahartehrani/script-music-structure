from NumberType import NumberType, get_number_type_by_name
import numpy as np

class HalfMatrix:
    @staticmethod
    def get_number_type():
        return NumberType
    
    def __init__(self, options):
        self.size = options.get('size')
        self.feature_amount = options.get('feature_amount', 1)
        self.number_type = get_number_type_by_name(options.get("number_type", NumberType.FLOAT32))
        self.sample_duration = options.get('sample_duration', 1)
        self.length = ((self.size * self.size + self.size) // 2) * self.feature_amount

        if 'buffer' in options:
            self.data = np.zeros(options['buffer'], dtype=np.uint8)
            assert self.length == len(self.data)
        else:
            self.data = np.zeros(self.length, dtype=np.uint8)

    def fill_features(self, callback):
        for y in range(self.size):
            cells_before = ((y * y + y) // 2) * self.feature_amount
            for x in range(y + 1):
                for f in range(self.feature_amount):
                    self.data[cells_before + x * self.feature_amount + f] = callback(x, y, f)

    def fill_features_normalized(self, callback):
        for y in range(self.size):
            cells_before = ((y * y + y) // 2) * self.feature_amount
            for x in range(y + 1):
                for f in range(self.feature_amount):
                    self.data[cells_before + x * self.feature_amount + f] = callback(x, y, f) * self.number_type.value['scale']

    @staticmethod
    def from_(matrix, options=None):
        if options is None:
            options = {}
        feature_amount = matrix.feature_amount
        number_type = matrix.number_type
        sample_duration = matrix.number_type

        return HalfMatrix({
            'size': matrix.size, 
            'feature_amount': feature_amount, 
            'number_type': number_type, 
            'sample_duration': sample_duration
        })
    
    def fill_by_index(self, callback):
        for i in range(len(self.data) - 1, -1, -1): 
            self.data[i] = callback(i)

    def get_value(self, x, y, f=0):
        index = ((y * y + y) // 2) * self.feature_amount + x * self.feature_amount + f
        return self.data[index]

    def has_cell(self, x, y):
        return x <= y and y < self.size and x >= 0 and y >= 0
    
    def get_values_normalized(self, x, y):
        values = np.zeros(self.feature_amount, dtype=np.uint8)
        for o in range(self.feature_amount):
            index = ((y * y + y) // 2) * self.feature_amount + x * self.feature_amount + o
            values[o] = self.data[index] / self.number_type.value['scale']
        return values
    
    def get_value_normalized(self, x, y, f=0):
        index = ((y * y + y) // 2) * self.feature_amount + x * self.feature_amount + f
        return self.data[index] / self.number_type.value['scale']

