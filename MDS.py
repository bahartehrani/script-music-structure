from Matrix import Matrix
from HalfMatrix import HalfMatrix
import math
import numpy as np
import seedrandom

def get_mds_coordinates(
    distance_matrix: HalfMatrix,
    strategy="Classic"
):
    print("CalculatingMDS")

    learn_rates = [13]
    best = None
    best_loss = 1
    coords = None

    if strategy == "Classic":
        coords = classical_mds(distance_matrix.get_nested_array(), 2)

    elif strategy == "GDTries":
        learn_rates.extend([0.5, 1, 2, 4, 8])
        for rate in learn_rates:
            coords = get_mds_coordinates_with_gradient_descent_matrix(Matrix(distance_matrix.get_nested_array()), lr=rate)
            if coords.loss_per_step[-1] < best_loss:
                best = coords
                print("Best is", rate)
                best_loss = coords.loss_per_step[-1]
        coords = best.coordinates.data

    elif strategy == "GD":
        for rate in learn_rates:
            coords = get_mds_coordinates_with_gradient_descent_matrix(Matrix(distance_matrix.get_nested_array()), lr=rate)
            if math.isnan(coords.loss_per_step[-1]):
                learn_rates.append(learn_rates[-1] / 2)
                print("Changing rate to", learn_rates[-1])
            elif coords.loss_per_step[-1] < best_loss:
                best = coords
                print("Best is", rate)
                best_loss = coords.loss_per_step[-1]
        coords = best.coordinates.data

    return normalize_2d_coordinates(coords)

def classical_mds(distances, dimensions=2):
    
    # Square distances
    M = -0.5 * np.square(distances)

    # Double centre the rows/columns
    def mean(A):
        return np.sum(A) / len(A)

    row_means = mean(M)
    col_means = mean(M.T)
    total_mean = mean(row_means)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i][j] += total_mean - row_means[i] - col_means[j]

    # Take the SVD of the double-centred matrix, and return the
    # points from it
    U, S, Vt = np.linalg.svd(M)
    eigen_values = np.sqrt(S)

    return [np.dot(row, eigen_values)[:dimensions] for row in U]

def get_mds_coordinates_with_gradient_descent_matrix(distances, lr=7, max_steps=500, 
                                                     min_loss_difference=1e-9, momentum=0, log_every=0):
    num_coordinates = distances.shape[0]
    coordinates = get_initial_mds_coordinates(num_coordinates)

    loss_per_step = []
    accumulation = None

    for step in range(max_steps):
        loss = get_mds_loss(distances, coordinates)
        loss_per_step.append(loss)

        # Check if we should early stop.
        if len(loss_per_step) > 1:
            loss_prev = loss_per_step[-2]
            if abs(loss_prev - loss) < min_loss_difference or np.isnan(loss) or loss > 10:
                return {'coordinates': coordinates, 'loss_per_step': loss_per_step}

        if log_every > 0 and step % log_every == 0:
            print(f'Step: {step}, loss: {loss}')

        # Apply the gradient for each coordinate.
        for coord_index in range(num_coordinates):
            gradient = get_gradient_for_coordinate(distances, coordinates, coord_index)

            if momentum == 0 or accumulation is None:
                accumulation = gradient
            else:
                accumulation = accumulation * momentum + gradient

            update = accumulation * lr
            updated_coordinates = coordinates[coord_index] - update
            coordinates[coord_index] = updated_coordinates

    return {'coordinates': coordinates, 'loss_per_step': loss_per_step}

def get_initial_mds_coordinates(num_coordinates, dimensions=2, seed=1):
    rng = seedrandom.SeededRNG(seed)
    random_uniform = np.array([[rng.random() for _ in range(dimensions)] for _ in range(num_coordinates)])
    return random_uniform / np.sqrt(dimensions)

import numpy as np

def get_mds_loss(distances, coordinates):
    loss = 0
    normalizer = coordinates.shape[0] ** 2
    for coord_index1 in range(coordinates.shape[0]):
        for coord_index2 in range(coordinates.shape[0]):
            if coord_index1 == coord_index2:
                continue

            coord1 = coordinates[coord_index1, :]
            coord2 = coordinates[coord_index2, :]
            target = distances[coord_index1, coord_index2]
            predicted = np.linalg.norm(coord1 - coord2)
            loss += (target - predicted) ** 2 / normalizer

    return loss

def get_gradient_for_coordinate(distances, coordinates, coord_index):
    coord = coordinates[coord_index, :]
    normalizer = coordinates.shape[0] ** 2
    gradient = np.zeros((1, coord.shape[0]))

    for other_coord_index in range(coordinates.shape[0]):
        if coord_index == other_coord_index:
            continue

        other_coord = coordinates[other_coord_index, :]
        squared_difference_sum = np.sum((coord - other_coord) ** 2)
        predicted = np.sqrt(squared_difference_sum)
        targets = [distances[coord_index, other_coord_index], distances[other_coord_index, coord_index]]

        for target in targets:
            loss_wrt_predicted = (-2 * (target - predicted)) / normalizer
            predicted_wrt_squared_difference_sum = 0.5 / np.sqrt(squared_difference_sum)
            squared_difference_sum_wrt_coord = 2 * (coord - other_coord)
            loss_wrt_coord = squared_difference_sum_wrt_coord * (loss_wrt_predicted * predicted_wrt_squared_difference_sum)
            gradient += loss_wrt_coord

    return gradient

import numpy as np

def normalize_2d_coordinates(coords):
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    radii = np.linalg.norm(centered_coords, axis=1)
    max_radius = np.max(radii)
    normalized_coords = centered_coords / max_radius

    return normalized_coords

def get_mds_feature(distance_matrix: HalfMatrix):
    feature = classical_mds(distance_matrix.get_nested_array(), 1)
    feature = np.array(feature)[:, 0]

    # Normalize to [0, 1]
    max_val = np.max(feature)
    min_val = np.min(feature)
    feature = (feature - min_val) / (max_val - min_val)

    return feature

def get_angle_and_radius(mds_coordinate):
    x = mds_coordinate[0]
    y = mds_coordinate[1]

    angle = math.atan2(y, x) / (2 * math.pi)
    angle = angle if angle >= 0 else 1 + angle

    radius = math.sqrt(x**2 + y**2)

    return angle, radius
