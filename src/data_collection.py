import requests
import pandas as pd

# Spotify API credentials
CLIENT_ID = 'your_client_id'
CLIENT_SECRET = 'your_client_secret'

def get_access_token(client_id, client_secret):
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

# Example usage
if __name__ == "__main__":
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    track_id = '3n3Ppam7vgaVa1iaRUc9Lp'  # Example track ID
    audio_features = get_audio_features(track_id, access_token)
    print(audio_features)

    # Example of organizing data into a DataFrame
    data = {
        'track_id': [track_id],
        'energy': [audio_features['energy']],
        'valence': [audio_features['valence']],
        'tempo': [audio_features['tempo']],
        'danceability': [audio_features['danceability']]
    }
    df = pd.DataFrame(data)
    print(df)
