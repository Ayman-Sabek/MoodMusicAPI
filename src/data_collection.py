import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from dotenv import load_dotenv
import os
import time

def authenticate_spotify():
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise Exception("Spotify API credentials are not set in the environment variables.")
    
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def get_audio_features(sp, track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 50):  # Spotify API allows max 50 track IDs per request
        batch = track_ids[i:i+50]
        features = sp.audio_features(batch)
        audio_features.extend(features)
    return audio_features

def main():
    sp = authenticate_spotify()
    
    # Search for tracks to get more diverse data
    queries = ["pop", "rock", "jazz", "classical", "hip hop", "electronic", "country", "blues", "reggae"]
    track_ids = []
    
    for query in queries:
        results = sp.search(q=query, limit=50, type='track')
        for item in results['tracks']['items']:
            track_ids.append(item['id'])
    
    audio_features = get_audio_features(sp, track_ids)
    df = pd.DataFrame(audio_features)
    
    if not df.empty:
        df.to_csv('extended_music_data.csv', index=False)
        print("Data collection completed and saved to extended_music_data.csv")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
