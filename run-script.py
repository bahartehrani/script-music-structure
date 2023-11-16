import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util
import numpy as np
from HalfMatrix import HalfMatrix
import DistanceCalc
import logging as log
from Features import Features
import Filter as Filter
import SSM as SSM
from Matrix import Matrix
import Structure as Structure
import math
import KeyDetection as KeyDetection

############# Spotify client setup: #############

client_id = ""
client_secret = ""
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
track_uri = 'spotify:track:2agJG5mKBoOL8uojjKQvCD'
track_id = track_uri.split(':')[-1]

audio_analysis = sp.audio_analysis(track_id)

############# Start with Self-Similarity Matrix computation: #############

segments = audio_analysis['segments']
num_segments = len(segments)

def calculate_ssm(features, sample_duration, all_pitches=False, threshold=0, similarity_function="cosine"):
    ssm = HalfMatrix({
        'size': len(features),
        'number_type': 'UINT8',
        'sample_duration': sample_duration,
        'feature_amount': 12 if all_pitches else 1
    })
    
    def fill_feature_function(x, y, f):
        if similarity_function == "cosine":
            return max(0, DistanceCalc.cosine_transposed(features[x], features[y], f) - threshold) / (1 - threshold)
        if similarity_function == "euclidean":
            val = DistanceCalc.euclidian_pitch_transposed(features[x], features[y], f)
            if val < 0 or val > 1:
                log.debug(x, y, val, features[x], features[y])
            return max(0, min(1, val - threshold)) / (1 - threshold)
    
    ssm.fill_features_normalized(fill_feature_function)
    return ssm


def compute_harmonic_structure(options):
    pitch_features = options.get('pitch_features')
    smoothed_features = Filter.gaussian_blur_features(pitch_features, 1)
    pitch_ssm = calculate_ssm(smoothed_features, options.get('sample_duration'), options.get('all_pitches'), 0.35, "euclidean")

    enhanced_ssm = SSM.enhance_ssm(
    pitch_ssm,
    {
        'blur_length': options.get('enhance_blur_length'),
        'tempo_ratios': options.get('tempo_ratios'),
        'strategy': 'linmed'
    }
    )
    transposition_invariant_pre = SSM.make_transposition_invariant(enhanced_ssm)
    strict_path_matrix_half = SSM.row_column_auto_threshold(transposition_invariant_pre, 0.15)
    strict_path_matrix = Matrix.from_half_matrix(strict_path_matrix_half)
    
    sample_amount = len(pitch_features)
    sample_duration = options.get('sample_duration')
    duration = 3
    sampled_segments = Structure.create_fixed_duration_structure_segments(sample_amount, sample_duration, duration)

    def update_callback(harmonic_structure, state="processing", strategy="Classic"):
        log.debug("Harmonic Structure", harmonic_structure)
        sorted_harmonic_structure_mds = color_harmonic_structure(harmonic_structure, strict_path_matrix, strategy)

        for section in sorted_harmonic_structure_mds:
            start_in_samples = math.floor(section.start / sample_duration)
            end_in_samples = math.floor(section.end / sample_duration)
            key = KeyDetection.detect(pitch_features, start_in_samples, end_in_samples)
            section.key = key

    harmonic_structure, mutor_group_amount, segments_mutor = Structure.find_mute_decomposition(
        strict_path_matrix,
        sampled_segments,
        sample_duration,
        "classic",
        "or",
        update_callback
    )

    update_callback(harmonic_structure, "done", "GD")

def color_harmonic_structure(harmonic_structure, ssm, strategy):
    log.debug(strategy)
    harmonic_structure_mds = Structure.mds_color_segments(harmonic_structure, ssm, "DTW", strategy, True)

    sorted_harmonic_structure_mds = sorted(harmonic_structure_mds, key=lambda x: (x.group_id, x.start))
    for i in range(0,10):
        print(sorted_harmonic_structure_mds[i].end)
    return sorted_harmonic_structure_mds


def process():
    features = Features(audio_analysis, {
        'samples': 600,
        'sample_duration': 0.33,
        'sample_blur': 1,
    })

    compute_harmonic_structure({
        'pitch_features': features.sampled.get('pitches'),
        'sample_duration': features.sample_duration,
        'all_pitches': False,
        'enhance_blur_length': 6,
        'tempo_ratios': [0.66, 0.81, 1, 1.22, 1.5],
    })


process()




