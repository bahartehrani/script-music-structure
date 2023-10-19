import math

def gaussian_blur_features(features, size):
    blurred_features = []
    full_kernel_size = size * 2 + 1
    kernel = generate_1d_gaussian_kernel(full_kernel_size, size / 2)
    feature_amount = len(features[0])

    for i in range(len(features)):
        new_vector = [0] * feature_amount
        for f in range(feature_amount):
            sum_ = 0
            kernel_sum = 0
            for k in range(-size, size+1):
                if 0 <= i + k < len(features):
                    sum_ += features[i + k][f] * kernel[k + size]
                    kernel_sum += kernel[k + size]
            new_vector[f] = sum_ / kernel_sum
        blurred_features.append(new_vector)

    return blurred_features

def generate_1d_gaussian_kernel(size, sigma=None):
    if sigma is None:
        sigma = size / 2

    kernel = [0.0] * size
    mean_index = (size - 1) / 2
    sum_ = 0  # For accumulating the kernel values

    for x in range(size):
        kernel[x] = math.exp(-0.5 * math.pow((x - mean_index) / sigma, 2.0))
        # Accumulate the kernel values
        sum_ += kernel[x]

    # Normalize the kernel
    for x in range(size):
        kernel[x] /= sum_

    return kernel
