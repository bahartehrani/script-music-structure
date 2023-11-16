from HalfMatrix import HalfMatrix
from Matrix import Matrix
import numpy as np

def get_distance_matrix(segments, path_SSM: Matrix, strategy, kappa=0.7):
    amount = len(segments)
    distance_matrix = HalfMatrix(size=amount, number_type=HalfMatrix.NumberType.FLOAT32)

    segment_induced_segments = []

    for segment in segments:
        sample_start = int(segment.start / path_SSM.sample_duration)
        sample_end = int(segment.end / path_SSM.sample_duration)
        induced_segments = get_induced_segments_from_sample_range(sample_start, sample_end, path_SSM)
        segment_induced_segments.append(induced_segments)

    def compute_distance(x, y):
        dist = 0
        same_group = segments[x].group_id == segments[y].group_id
        if strategy == "DTW":
            dist = (1 - segment_similarity_DTW(segments[x], segments[y], path_SSM)) * (kappa if same_group else 1)
        elif strategy == "overlap":
            dist = segment_distance_overlap(segment_induced_segments[x], segment_induced_segments[y])
        return dist

    distance_matrix.fill(compute_distance)

    return distance_matrix

def get_induced_segments_from_sample_range(start, end, path_SSM):
    D, width, height, score = compute_accumulated_score_matrix(path_SSM, start, end)
    path_family = compute_optimal_path_family(D, width, height)
    induced_segments = get_induced_segments(path_family)
    return induced_segments

def get_induced_segments(path_family):
    path_amount = len(path_family)
    induced_segments = np.zeros(path_amount * 2, dtype=np.uint16)

    if path_amount > 0:
        for p in range(path_amount):
            # paths stored in reverse due to backtracking
            path_end_y = path_family[p][1][1]  # accessing the y value of the tuple
            path_start_y = path_family[p][-1][1]  # accessing the y value of the last tuple
            induced_segments[p * 2] = path_start_y
            induced_segments[p * 2 + 1] = path_end_y

    return induced_segments


# The ratio between the length of the knight move vs length of a diagonal move
knight_move_ratio = 1  # np.sqrt(10) / 2 + .01  # plus slight offset to favour diagonal moves when going through penalties
knight_move_tweak = 1  # 0.99  # Also favouring diagonal moves when accumulating score

def compute_accumulated_score_matrix(ssm, start, end, D=None):
    sample_amount = ssm.height
    if start < 0:
        print("Error: start below 0: ", start)
    if end > sample_amount:
        print("Error: end above sample_amount: ", sample_amount, "end", end)

    # print('tbahar')
    # print(end)
    # print(start)
    length = end - start

    width = length + 1
    height = sample_amount
    
    # accumulatedScoreMatrix length + 1 for elevator
    if D is None:
        D = np.zeros(height * width, dtype=np.float32)
        D.fill(np.NINF)

    penalty = -2

    def penalize(value):
        return penalty if value <= 0 else value

    D[0] = 0
    D[1] = penalize(ssm.get_value_normalized(start, 0))

    for y in range(1, height):
        D[y * width + 0] = max(D[(y - 1) * width + 0], D[(y - 1) * width + width - 1])
        D[y * width + 1] = D[y * width + 0] + penalize(ssm.get_value_normalized(start, y))
        for x in range(2, width):
            down = np.NINF if y == 1 else D[(y - 2) * width + x - 1]
            right = D[(y - 1) * width + x - 2]
            diag = D[(y - 1) * width + x - 1]
            D[y * width + x] = penalize(ssm.get_value_normalized(start + x - 1, y)) + max(diag, right, down)

    score = max(D[(height - 1) * width + 0], D[(height - 1) * width + width - 1])
    # print(score)
    return {"D": D, "width": width, "height": height, "score": score}


def segment_similarity_DTW(segmentA, segmentB, ssm: Matrix):
    start_sample_A = int(segmentA.start / ssm.sample_duration)
    end_sample_A = int(segmentA.end / ssm.sample_duration)
    height = end_sample_A - start_sample_A + 1

    start_sample_B = int(segmentB.start / ssm.sample_duration)
    end_sample_B = int(segmentB.end / ssm.sample_duration)
    width = end_sample_B - start_sample_B + 1

    D = np.zeros(height * width, dtype=np.float32)
    D.fill(np.NINF)
    
    penalty = 0
    
    def penalize(value):
        return penalty if value <= 0 else value

    D[0] = penalize(ssm.get_value_normalized(start_sample_A, start_sample_B))

    for y in range(height):
        for x in range(width):
            if y == 0 and x == 0:
                continue
            down = D[(y - 2) * width + x - 1] if y - 2 >= 0 and x - 1 >= 0 else np.NINF
            right = D[(y - 1) * width + x - 2] if y - 1 >= 0 and x - 2 >= 0 else np.NINF
            diag = D[(y - 1) * width + x - 1] if y - 1 >= 0 and x - 1 >= 0 else np.NINF
            
            D[y * width + x] = penalize(ssm.get_value_normalized(start_sample_B + x, start_sample_A + y)) + max(diag, right, down)

    score = D[-1]
    if score <= 0:
        return 0

    x, y = width - 1, height - 1
    path_length = 1
    while x > 0 or y > 0:
        down = D[(y - 2) * width + x - 1] if y - 2 >= 0 and x - 1 >= 0 else np.NINF
        right = D[(y - 1) * width + x - 2] if y - 1 >= 0 and x - 2 >= 0 else np.NINF
        diag = D[(y - 1) * width + x - 1] if y - 1 >= 0 and x - 1 >= 0 else np.NINF

        if x == 0:
            x -= 1
        elif y == 0:
            y -= 1
        elif down >= right and down >= diag:
            y -= 2
            x -= 1
        elif right >= down and right >= diag:
            y -= 1
            x -= 2
        elif diag >= down and diag >= right:
            y -= 1
            x -= 1
        path_length += 1
    
    similarity = score / path_length

    return similarity

def compute_optimal_path_family(D, width, height):
    path_family = []
    path = []

    y = height - 1
    if D[y * width + width - 1] < D[y * width]:
        x = 0
    else:
        x = width - 1
        path.append(x - 1)
        path.append(y)

    predecessors = np.zeros(6, dtype=np.uint16)
    predecessor_length = 0  # in pairs, so max would be 3

    while y > 0 or x > 0:
        # obtaining the set of possible predecessors given our current position
        predecessors[0] = y - 1
        predecessors[1] = x - 1
        if y <= 2 and x <= 2:
            predecessor_length = 1
        elif y <= 2 and x > 2:
            predecessors[2] = y - 1
            predecessors[3] = x - 2
            predecessor_length = 2
        elif y > 2 and x <= 2:
            predecessors[2] = y - 2
            predecessors[3] = x - 1
            predecessor_length = 2
        else:
            predecessors[2] = y - 2
            predecessors[3] = x - 1
            predecessors[4] = y - 1
            predecessors[5] = x - 2
            predecessor_length = 3

        if y == 0:
            x -= 1
        elif x == 0:
            if D[(y - 1) * width + width - 1] > D[(y - 1) * width]:
                y -= 1
                x = width - 1
                if path:
                    path_family.append(path)
                path = [x - 1, y]
            else:
                y -= 1
                x = 0
        elif x == 1:
            x = 0
        else:
            max_val = np.NINF
            for i in range(predecessor_length):
                val = D[predecessors[i * 2] * width + predecessors[i * 2 + 1]]
                if val > max_val:
                    max_val = val
                    y = predecessors[i * 2]
                    x = predecessors[i * 2 + 1]
            path.append(x - 1)
            path.append(y)

    path_family.append(path)
    return path_family


def segment_distance_overlap(x, y):
    pass

def create_score_matrix_buffer(sample_amount):
    D = np.zeros(sample_amount * sample_amount, dtype=np.float32)
    return D.fill(np.NINF)

logi = 0
def compute_segment_path_family_info(path_ssm, start_in_samples, end_in_samples, score_matrix_buffer=None, strategy=None):
    global logi
    logi += 1

    sample_amount = path_ssm.height
    if not score_matrix_buffer:
        score_matrix_buffer = create_score_matrix_buffer(sample_amount)

    P, path_scores, score, width = extract_path_family(path_ssm, start_in_samples, end_in_samples)

    def fitness_function(strategy, P, path_scores, score, sample_amount, width):
        return compute_fitness(P, path_scores, score, sample_amount, width)

    fitness_data = fitness_function(strategy, P, path_scores, score, sample_amount, width)
    
    path_family = []
    for path in fitness_data['pruned_path_family']:
        path_coords = []
        for i in range(0, len(path), 2):
            x = start_in_samples + path[i]
            y = path[i + 1]
            path_coords.append([x, y])
        path_family.append(path_coords)

    return {
        'score': score,
        'path_scores': path_scores,
        'normalized_score': fitness_data['normalized_score'],
        'coverage': fitness_data['coverage'],
        'normalized_coverage': fitness_data['normalized_coverage'],
        'fitness': fitness_data['fitness'],
        'path_family': path_family
    }

def compute_induced_coverage(path_family):
    path_amount = len(path_family)
    coverage = 0

    if path_amount > 0:
        for p in path_family:
            path_end_y = p[1]
            path_start_y = p[-1]
            coverage += abs(path_end_y - path_start_y)

    return coverage

def compute_fitness(path_family, path_scores, score, sample_amount, width):
    error = 1e-16

    # Normalized score
    # We subtract the given self similarity path, and divide by total length of all paths (+ error to prevent divide by 0)
    path_family_length = sum(len(p) / 2 for p in path_family)  # /2 because we store x and y flat
    normalized_score = max(0, (score - width) / (path_family_length + error))

    # Normalized coverage
    coverage = compute_induced_coverage(path_family)
    normalized_coverage = (coverage - width) / (sample_amount + error)

    # Fitness
    fitness = (2 * normalized_score * normalized_coverage) / (normalized_score + normalized_coverage + error)

    return {
        "fitness": fitness,
        "normalized_score": normalized_score,
        "coverage": coverage,
        "normalized_coverage": normalized_coverage,
        "path_family_length": path_family_length,
        "pruned_path_family": path_family
    }

def extract_path_family(ssm, start, end):
    accumulated_score_matrix = compute_accumulated_score_matrix(ssm, start, end)
    D = accumulated_score_matrix.get('D')
    width = accumulated_score_matrix.get('width')
    height = accumulated_score_matrix.get('height')
    score = accumulated_score_matrix.get('score')
    path_family = compute_optimal_path_family(D, width, height)
    path_scores = get_brightness_for_path_family(ssm, path_family, start)

    return path_family, path_scores, score, width

deb = 0

def get_brightness_for_path_family(pathSSM, path_family, start):
    global deb
    deb += 1
    path_scores = []
    for path in path_family:
        total = 0
        for i in range(0, len(path), 2):
            x = start + path[i + 0]
            y = path[i + 1]
            val = pathSSM.get_value_normalized(x, y)
            total += val
        average = total / (len(path) / 2)
        path_scores.append(average)
    return path_scores
