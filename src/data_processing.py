import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
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

# Example usage
if __name__ == "__main__":
    file_path = 'path_to_your_data.csv'  # Replace with your actual file path
    df = load_data(file_path)
    if df is not None:
        processed_df = clean_and_process_data(df)
        print(processed_df.head())  # Display the first few rows of the processed DataFrame
