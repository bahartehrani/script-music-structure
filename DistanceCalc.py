def squared_distance_transposed(a, b, p):
    dist = 0
    for i in range(len(a)):
        transposed_i = (i + p) % 12
        diff = a[i] - b[transposed_i]
        dist += diff * diff
    return dist

def cosine_transposed(a, b, p):
    adotv = 0
    amag = 0
    bmag = 0
    for i in range(len(a)):
        transposed_i = (i + p) % 12
        adotv += a[i] * b[transposed_i]
        amag += a[i] * a[i]
        bmag += b[transposed_i] * b[transposed_i]
    
    amag = amag ** 0.5
    bmag = bmag ** 0.5
    return adotv / (amag * bmag)

def euclidian_distance_transposed(a, b, p):
    return squared_distance_transposed(a, b, p) ** 0.5

def euclidian_pitch_transposed(a, b, p):
    return 1 - euclidian_distance_transposed(a, b, p) / (12 ** 0.5)

    