from HalfMatrix import HalfMatrix
import math
import numpy as np

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

