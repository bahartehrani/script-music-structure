from NumberType import NumberType, get_number_type_by_name

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
            self.data = list(options['buffer'])
            assert self.length == len(self.data)
        else:
            self.data = [0] * self.length

    def fill_features_normalized(self, callback):
        for y in range(self.size):
            cells_before = ((y * y + y) // 2) * self.feature_amount
            for x in range(y + 1):
                for f in range(self.feature_amount):
                    self.data[cells_before + x * self.feature_amount + f] = callback(x, y, f) * self.number_type.scale