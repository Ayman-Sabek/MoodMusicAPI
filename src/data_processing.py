import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The CSV file is empty.")
        print(f"Data loaded successfully with {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def clean_and_process_data(df):
    """
    Clean and process the data.
    """
    # Drop rows with missing values
    df.dropna(inplace=True)
    print(f"Data after dropping missing values: {len(df)} records.")

    # Convert data types
    df['energy'] = df['energy'].astype(float)
    df['valence'] = df['valence'].astype(float)
    df['tempo'] = df['tempo'].astype(float)
    df['danceability'] = df['danceability'].astype(float)

    # Normalize numerical features (optional)
    df['energy'] = (df['energy'] - df['energy'].min()) / (df['energy'].max() - df['energy'].min())
    df['valence'] = (df['valence'] - df['valence'].min()) / (df['valence'].max() - df['valence'].min())
    df['tempo'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
    df['danceability'] = (df['danceability'] - df['danceability'].min()) / (df['danceability'].max() - df['danceability'].min())

    print("Data processing completed.")
    return df

def feature_engineering(df):
    """
    Perform feature engineering to create new features.
    """
    # Create a new feature based on existing ones
    df['mood_score'] = (df['energy'] + df['valence']) / 2

    # Extracting parts of date (assuming 'release_date' is in YYYY-MM-DD format)
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year
    df['release_month'] = pd.to_datetime(df['release_date']).dt.month
    df['release_day'] = pd.to_datetime(df['release_date']).dt.day

    # Create interaction terms
    df['energy_danceability'] = df['energy'] * df['danceability']

    # Log transformation (if needed)
    df['log_tempo'] = df['tempo'].apply(lambda x: np.log(x + 1))

    print("Feature engineering completed.")
    return df

# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    file_path = 'music_data.csv'  # Ensure this is the correct path to your CSV file
    df = load_data(file_path)
    if df is not None:
        processed_df = clean_and_process_data(df)
        engineered_df = feature_engineering(processed_df)
        print(engineered_df.head())  # Display the first few rows of the DataFrame with new features
