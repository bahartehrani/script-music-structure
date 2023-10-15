import math

# Constants
circle_of_fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # Starting from C=0
note_names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
key_names = [
    "C",
    "D♭",
    "D",
    "E♭",
    "E",
    "F",
    "F♯",
    "G",
    "A♭",
    "A",
    "B♭",
    "B",
    "Cm",
    "D♭m",
    "Dm",
    "E♭m",
    "Em",
    "Fm",
    "F♯m",
    "Gm",
    "A♭m",
    "Am",
    "B♭m",
    "Bm",
]

# DEAL WITH D3 REPLACEMENT:
# Placeholder for d3 color scale since there's no direct equivalent in Python
# For a complete translation, you'd need a different library or method to handle color interpolation
def color_wheel(angle):
    return "#000000"  # Return black as a placeholder

def get_note_name(i):
    return note_names[i]

DEGREES_PER_RADIAN = 180 / math.pi
RADIANS_PER_DEGREE = math.pi / 180
TWO_PI = 2 * math.pi
OFFSET = math.pi / 2  # (3 * math.pi) / 2; // full cycle is 2pi

def sort_with_indices(to_sort):
    to_sort_val_index = list(enumerate(to_sort))
    to_sort_val_index.sort(key=lambda x: x[1])
    return [i[0] for i in to_sort_val_index]

def tonality(pitches):
    x = 0
    y = 0
    energy = 0
    for i in range(12):
        vangle = (circle_of_fifths[i] / 12.0) * TWO_PI
        vradius = pitches[i]  # Between 0 and 1
        energy += vradius / 12
        x += vradius * math.cos(vangle)
        y += vradius * math.sin(vangle)
    angle = (1 + math.atan2(y, x) / TWO_PI) % 1
    radius = math.sqrt(x**2 + y**2) / (energy * 12)
    return angle, radius, energy

# ... (Additional methods would go here, like tonalityThirds, tonalityThird, tonalVectorColor, etc.)

def loudness(db):
    return max(0, 60 + db) / 60

def loudness_perceived(db):
    return 2**(db / 10)

def log_compression(value, gamma=1):
    return math.log(1 + gamma * value) / math.log(gamma)

def tonality_thirds(pitches):
    x = 0
    y = 0
    energy = 0

    sorted_pitch_indexes = sort_with_indices(pitches)[::-1]
    for i in range(len(pitches)):
        index = sorted_pitch_indexes[i]
        if index == -1:
            continue
        sorted_pitch_indexes[i] = -1
        vangle = -(circle_of_fifths[index] / 12.0) * TWO_PI + OFFSET
        vradius = pitches[index]  # Between 0 and 1
        energy += vradius / 12
        x += vradius * math.cos(vangle)
        y += vradius * math.sin(vangle)

        maj_third_index = (index + 4) % 12
        if maj_third_index in sorted_pitch_indexes:
            vangle = -(((12 + circle_of_fifths[maj_third_index] - 3.5) % 12) / 12.0) * TWO_PI + OFFSET
            vradius = pitches[maj_third_index]  # Between 0 and 1
            energy += vradius / 12
            x += vradius * math.cos(vangle)
            y += vradius * math.sin(vangle)
            sorted_pitch_indexes[sorted_pitch_indexes.index(maj_third_index)] = -1

    angle = (1 - math.atan2(x, y) / TWO_PI + 0.25) % 1
    radius = math.sqrt(x**2 + y**2) / (energy * 12)
    return angle, radius, energy


def tonality_third(pitches):
    x = 0
    y = 0
    energy = 0

    max_index = -1
    max_val = -1
    second_max_index = -1
    second_max = -1

    for i in range(12):
        if pitches[i] > max_val:
            max_val = pitches[i]
            max_index = i
        if pitches[i] >= second_max and i != max_index:
            second_max = pitches[i]
            second_max_index = i

    # major third apart
    if (12 + second_max_index - max_index) % 12 != 4:
        second_max_index = -1

    for i in range(12):
        vangle = -(circle_of_fifths[i] / 12.0) * TWO_PI + OFFSET
        if i == second_max_index:
            # add differently
            vangle = -(((12 + circle_of_fifths[i] - 3.5) % 12) / 12.0) * TWO_PI + OFFSET
        vradius = pitches[i]  # Between 0 and 1
        energy += vradius / 12
        x += vradius * math.cos(vangle)
        y += vradius * math.sin(vangle)

    angle = (1 - math.atan2(x, y) / TWO_PI + 0.25) % 1
    radius = math.sqrt(x**2 + y**2) / (energy * 12)
    return angle, radius, energy


def tonal_vector_color(pitches):
    angle, radius, energy = tonality(pitches)
    # Placeholder for color interpolation, you may want to replace this with actual color interpolation in Python.
    color = color_wheel(angle)
    saturation = min((radius * 3) / (energy * 12), 1)
    # Assuming some kind of color class where saturation is a property; you'd use a different method for Python color libraries
    color.s = saturation
    return color.hex()  # or return a suitable representation of the color

