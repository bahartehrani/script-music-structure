import spotipy
import logging as log
import scipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import FeaturesOld

############# Spotify client setup: #############

client_id = "18b62f10adae4a23a7824e9dc9f80da9"
client_secret = "c5192bb99a7044ab8d349be6dd80de49"
username = "noteaholic"

scope = "user-library-read"
redirect_uri = "http://localhost:8080/callback"
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
if token: 
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

# Start with 1 track:
track_uri = 'spotify:track:2agJG5mKBoOL8uojjKQvCD?si=9ec0419ca0fe4313'
track_id = track_uri.split(':')[-1]
track_info = sp.track(track_uri)

audio_analysis = sp.audio_analysis(track_id)

############# Start with Self-Similarity Matrix computation: #############

segments = audio_analysis['segments']
num_segments = len(segments)

self_similarity_matrix = np.zeros((num_segments, num_segments))

class HalfMatrix:
    def __init__(self, options):
        self.size = options.get('size')
        self.feature_amount = options.get('feature_amount', 1)
        self.number_type = 'FLOAT32'
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


def calculate_ssm(features, sample_duration, all_pitches=False, threshold=0, similarity_function="cosine"):
    ssm = HalfMatrix(
        size=len(features),
        number_type='UINT8',
        sample_duration=sample_duration,
        feature_amount=12 if all_pitches else 1
    )

    def euclidian_pitch_transposed(a, b, p):
        return 1 - euclidian_distance_transposed(a, b, p) / (12 ** 0.5)

    def euclidian_distance_transposed(a, b, p):
        return squared_distance_transposed(a, b, p) ** 0.5

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

    def fill_feature_function(x, y, f):
        if similarity_function == "cosine":
            return max(0, cosine_transposed(features[x], features[y], f) - threshold) / (1 - threshold)
        if similarity_function == "euclidean":
            val = euclidian_pitch_transposed(features[x], features[y], f)
            if val < 0 or val > 1:
                log.debug(x, y, val, features[x], features[y])
            return max(0, min(1, val - threshold)) / (1 - threshold)

    ssm.fill_features_normalized(fill_feature_function)
    return ssm

############# Compute sample pitches through Features class: #############

# Don't need a Track class yet, we can just directly pass the audio features into the Features class

def compute_harmonic_structure():
    # TODO: implement
    pass

def process(self):
    self.processing = True
    
    self.features = FeaturesOld(self.analysisData, {
        'samples': 600,
        'sampleDuration': 0.33,
        'sampleBlur': 1,
    })

    compute_harmonic_structure()

    self.processed = True
    self.processing = False


# Need to create Features class and process in constructor
# Then compute harmonic structure using sampled pitches
# Then compute self-similarity matrix