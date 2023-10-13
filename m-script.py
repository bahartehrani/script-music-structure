import spotipy
import scipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

client_id = "CLIENT ID"
client_secret = "CLIENT SECRET"
username = "USERNAME"

scope = "user-library-read"
redirect_uri = "CALLBACK"
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
if token: 
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

# Start with 1 track:
track_uri = 'spotify:track:TRACK_ID'
track_id = track_uri.split(':')[-1]
track_info = sp.track(track_uri)

audio_analysis = sp.audio_analysis(track_id)

############# Start with Self-Similarity Matrix computation: #############

segments = audio_analysis['segments']
num_segments = len(segments)

self_similarity_matrix = np.zeros((num_segments, num_segments))

class HalfMatrix:
    class NumberType:
        UINT8 = 'UINT8'

    def __init__(self, options):
        self.size = options.get('size')
        self.feature_amount = options.get('feature_amount', 1)
        self.number_type = get_number_type_by_name(options.get('number_type', NumberType.FLOAT32))
        self.sample_duration = options.get('sample_duration', 1)
        self.length = ((self.size * self.size + self.size) // 2) * self.feature_amount

        if 'buffer' in options:
            # In python, instead of using JavaScript typed arrays like Float32Array, you might use numpy arrays.
            # Here's a placeholder using a list, but consider replacing with an appropriate numpy datatype.
            self.data = list(options['buffer'])
            assert self.length == len(self.data)
        else:
            self.data = [0] * self.length  # Again, consider using numpy for actual numerical computation.

def calculate_ssm(features, sample_duration, all_pitches=False, threshold=0, similarity_function="cosine"):
    ssm = HalfMatrix(
        size=len(features),
        number_type=HalfMatrix.NumberType.UINT8,
        sample_duration=sample_duration,
        feature_amount=12 if all_pitches else 1
    )

    def fill_feature_function(x, y, f):
        if similarity_function == "cosine":
            return max(0, sim.cosine_transposed(features[x], features[y], f) - threshold) / (1 - threshold)
        elif similarity_function == "euclidean":
            val = sim.euclidian_pitch_transposed(features[x], features[y], f)
            # Assuming 'log' is a logger you'll have to import or define.
            if val < 0 or val > 1:
                log.debug(x, y, val, features[x], features[y])
            return max(0, min(1, val - threshold)) / (1 - threshold)
        elif similarity_function == "euclideanTimbre":
            return max(0, min(1, sim.euclidian_timbre(features[x], features[y]) - threshold)) / (1 - threshold)

    ssm.fill_features_normalized(fill_feature_function)
    return ssm





