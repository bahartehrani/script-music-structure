import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util

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

