import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_access_token():
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })
    auth_response_data = auth_response.json()
    return auth_response_data['access_token']

def get_audio_features(track_id, access_token):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    audio_features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    response = requests.get(audio_features_url, headers=headers)
    return response.json()

def get_tracks_from_playlist(playlist_id, access_token):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    playlist_tracks_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    response = requests.get(playlist_tracks_url, headers=headers)
    return response.json()

# Example usage
if __name__ == "__main__":
    access_token = get_access_token()
    playlist_id = '37i9dQZF1DXcBWIGoYBM5M'  # Example playlist ID
    playlist_tracks = get_tracks_from_playlist(playlist_id, access_token)

    # Collect audio features for each track
    track_features = []
    for item in playlist_tracks['items']:
        track = item['track']
        track_id = track['id']
        audio_features = get_audio_features(track_id, access_token)
        track_data = {
            'track_id': track_id,
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'energy': audio_features['energy'],
            'valence': audio_features['valence'],
            'tempo': audio_features['tempo'],
            'danceability': audio_features['danceability']
        }
        track_features.append(track_data)

    df = pd.DataFrame(track_features)
    print(df)
