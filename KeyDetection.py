import math
import numpy as np
import AudioUtil as AudioUtil
import DistanceCalc as DistanceCalc

major_profile_temperley_norm = [1, 0.4, 0.7, 0.4, 0.9, 0.8, 0.4, 0.9, 0.4, 0.7, 0.3, 0.8]
minor_profile_temperley_norm = [1, 0.4, 0.7, 0.9, 0.4, 0.8, 0.4, 0.9, 0.7, 0.4, 0.3, 0.8]
major_profile_temperley_centered = [
    1.7917,
    -1.2083,
    0.2917,
    -1.2083,
    1.2917,
    0.7917,
    -1.2083,
    1.2917,
    -1.2083,
    0.2917,
    -1.7083,
    0.7917,
]
minor_profile_temperley_centered = [
    1.7917,
    -1.2083,
    0.2917,
    1.2917,
    -1.2083,
    0.7917,
    -1.2083,
    1.2917,
    0.2917,
    -1.2083,
    -1.7083,
    0.7917,
]

def detect(pitch_features, start, end):
    length = end - start
    average_pitches = np.zeros(12)
    for i in range(start, end):
        for p in range(12):
            average_pitches[p] += pitch_features[i][p] / length
    return detect_single(average_pitches)

w = 0

def detect_2D(pitches):
    global w
    correlation = correlate(pitches)
    w += 1
    if w % 200 == 0:
        print(correlation)  # assuming the log.debug function is equivalent to printing

    x, y, energy = 0, 0, 0
    for i in range(24):
        index = i if i < 12 else (i + 3) % 12
        vangle = (AudioUtil.circle_of_fifths(index) / 12.0) * (2 * math.pi)
        vradius = max(0, correlation[i])
        energy += abs(vradius) / 12
        x += vradius * math.cos(vangle)
        y += vradius * math.sin(vangle)

    angle = (1 + math.atan2(y, x) / (2 * math.pi)) % 1
    radius = math.sqrt(x**2 + y**2) / (energy * 12)
    return [angle, radius, energy]

def detect_single(pitches):
    correlation = correlate(pitches)
    max_val = np.NINF
    max_index = -1
    for index, val in enumerate(correlation):
        if val > max_val:
            max_val = val
            max_index = index
    return max_index

def get_name(key_index):
    return AudioUtil.key_names(key_index)

def profile_similarity(pitches):
    major_profile = major_profile_temperley_norm
    minor_profile = minor_profile_temperley_norm
    major_similarity = [DistanceCalc.cosine_transposed(pitches, major_profile, p) for p in range(12)]
    minor_similarity = [DistanceCalc.cosine_transposed(pitches, minor_profile, p) for p in range(12)]
    return major_similarity + minor_similarity

def correlate(pitches):
    major_profile = major_profile_temperley_centered
    minor_profile = minor_profile_temperley_centered

    pitches_average = sum(pitches) / 12
    sum_distance_major = [
        sum([(val - pitches_average) * major_profile[(12 + i - p) % 12] for i, val in enumerate(pitches)])
        for p in range(12)
    ]
    sum_distance_minor = [
        sum([(val - pitches_average) * minor_profile[(12 + i - p) % 12] for i, val in enumerate(pitches)])
        for p in range(12)
    ]
    abs_sum_distance_pitches = math.sqrt(sum([(val - pitches_average)**2 for i, val in enumerate(pitches)]))
    abs_sum_distance_major = math.sqrt(sum([val**2 for val in major_profile]))
    abs_sum_distance_minor = math.sqrt(sum([val**2 for val in minor_profile]))

    r_major = [val / (abs_sum_distance_pitches * abs_sum_distance_major) for val in sum_distance_major]
    r_minor = [val / (abs_sum_distance_pitches * abs_sum_distance_minor) for val in sum_distance_minor]

    return r_major + r_minor

def circle_of_fifths_angle(key_index):
    if key_index < 12:
        return AudioUtil.circle_of_fifths(key_index) / 12
    else:
        return AudioUtil.circle_of_fifths((key_index + 3) % 12) / 12
