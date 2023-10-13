import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util
import numpy as np
import matplotlib.pyplot as plt

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

# Start with Self-Similarity Matrix computation:

segments = audio_analysis['segments']
num_segments = len(segments)

self_similarity_matrix = np.zeros((num_segments, num_segments))

# Simple function to calculate similarity
def calculate_similarity(segment1, segment2):
    p1 = segment1['pitch']
    p2 = segment2['pitch']
    euclidean_distance = np.linalg.norm(p1 - p2)
    similarity = 1 / (1 + euclidean_distance)

    return similarity

# Create the SSM
for i in range(num_segments):
    for j in range(num_segments):
        similarity = calculate_similarity(segments[i], segments[j])
        self_similarity_matrix[i, j] = similarity



