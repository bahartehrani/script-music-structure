from HalfMatrix import HalfMatrix
import math
import numpy as np
from NumberType import NumberType

def enhance_ssm(ssm, options):
    blur_length = options.get('blur_length') or round(options.get('blur_time') / ssm.sample_duration) or 4
    tempo_ratios = options.get('tempo_ratios') or [1]

    enhancement_passes = []
    for tempo_ratio in tempo_ratios:
        enhancement_passes.append(
            median_smoothing(linear_smoothing(ssm, blur_length, tempo_ratio), blur_length * 1.5, tempo_ratio)
        )

    enhanced_ssm = HalfMatrix.from_(ssm) 
    def get_max_at_index(i):
        max_val = 0
        for ep in enhancement_passes:
            if ep.data[i] > max_val:
                max_val = ep.data[i]
        return max_val
    enhanced_ssm.fill_by_index(get_max_at_index)
    
    return enhanced_ssm

def linear_smoothing(ssm, length, tempo_ratio):
    smoothed_ssm = HalfMatrix.from_(ssm)

    blur = round(length / 2)
    total = pow(blur + 1, 2)
    tempos = np.zeros((blur * 2 + 1), dtype=np.int8)
    for i in range(-blur, 1 + blur):
        tempos[i + blur] = math.floor(i * tempo_ratio + 0.5)

    def feature_callback(x, y, f):
        sum_ = 0
        for i in range(-blur, 1 + blur):
            if ssm.has_cell(x + i, y + tempos[i + blur]):
                sum_ += ssm.get_value(x + i, y + tempos[i + blur], f) * (blur + 1 - abs(i))
        return sum_ / total

    smoothed_ssm.fill_features(feature_callback)
    return smoothed_ssm

def median_smoothing(ssm, length, tempo_ratio, resolution=128):
    buckets = np.zeros(resolution, dtype=np.float32)

    l = math.floor((length - 1) / 2)

    tempos = np.zeros((l * 2 + 1), dtype=np.int8)
    for i in range(-l, 1 + l):
        tempos[i + l] = math.floor(i * tempo_ratio + 0.5)
    
    smoothed_ssm = HalfMatrix.from_(ssm)

    def feature_callback_normalized(x, y, f):
        total_values = l * 2 + 1
        for offset in range(-l, l + 1):
            if ssm.has_cell(x + offset, y + tempos[offset + l]):
                value = ssm.get_value_normalized(x + offset, y + tempos[offset + l], f)
                buckets[math.floor(value * (resolution - 1))] += 1
            else:
                buckets[0] += 1
        middle = total_values / 2

        # Both check middle and clear buckets
        mean = -1
        for i in range(resolution):
            middle -= buckets[i]
            if middle < 0 and mean == -1:
                mean = i / (resolution - 1)
            buckets[i] = 0
        return mean

    smoothed_ssm.fill_features_normalized(feature_callback_normalized)
    
    return smoothed_ssm

def make_transposition_invariant(ssm: HalfMatrix):
    length_without_features = ssm.length // ssm.feature_amount 
    transposition_invariant_ssm = HalfMatrix({
        'size': ssm.size, 
        'number_type': 'UINT8',
        'sample_duration': ssm.sample_duration
    })

    i = 0
    while i < length_without_features:
        max_val = 0
        for f in range(ssm.feature_amount):
            if ssm.data[i * ssm.feature_amount + f] > max_val:
                max_val = ssm.data[i * ssm.feature_amount + f]
        transposition_invariant_ssm.data[i] = max_val
        i += 1
    # print(transposition_invariant_ssm.data[:100])
    return transposition_invariant_ssm


def row_column_auto_threshold(ssm: HalfMatrix, percentage_row, percentage_col=None):
    if percentage_col is None:
        percentage_col = percentage_row
    
    type_scale = ssm.number_type.value['scale']

    row_binary_matrix = HalfMatrix(size=ssm.size, number_type=NumberType.UINT8)
    col_binary_matrix = HalfMatrix(size=ssm.size, number_type=NumberType.UINT8)
    frequencies = np.zeros((type_scale + 1), dtype=np.uint16)

    for row in range(ssm.size):
        frequencies.fill(0)
        for col in range(ssm.size):
            frequencies[ssm.get_value_mirrored(col, row)] += 1
        stop_position = ssm.size * percentage_row
        threshold_value = 0
        for i in range(type_scale, 0, -1):
            stop_position -= frequencies[i]
            if stop_position <= 0:
                threshold_value = i
                break
        for col in range(row + 1):
            if ssm.get_value(col, row) >= threshold_value:
                value = min(
                    max(ssm.get_value(col, row) - threshold_value, 0) / (type_scale - threshold_value) * type_scale,
                    type_scale
                )
                row_binary_matrix.set_value(col, row, value)

    for col in range(ssm.size):
        frequencies.fill(0)
        for row in range(ssm.size):
            frequencies[ssm.get_value_mirrored(col, row)] += 1
        stop_position = ssm.size * percentage_col
        threshold_value = 0
        for i in range(type_scale, 0, -1):
            stop_position -= frequencies[i]
            if stop_position <= 0:
                threshold_value = i
                break
        for row in range(col, ssm.size):
            if ssm.get_value(col, row) >= threshold_value:
                value = min(
                    max(ssm.get_value(col, row) - threshold_value, 0) / (type_scale - threshold_value) * type_scale,
                    type_scale
                )
                col_binary_matrix.set_value(col, row, value)

    threshold_ssm = HalfMatrix.from_(ssm)
    threshold_ssm.fill(lambda x, y: (row_binary_matrix.get_value(x, y) + col_binary_matrix.get_value(x, y)) / 2)

    return threshold_ssm
