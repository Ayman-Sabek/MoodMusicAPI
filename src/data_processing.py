import pandas as pd

def clean_and_process_data(df):
    # Example of data cleaning and processing steps
    df.dropna(inplace=True)  # Remove rows with missing values
    df['energy'] = df['energy'].astype(float)
    df['valence'] = df['valence'].astype(float)
    df['tempo'] = df['tempo'].astype(float)
    df['danceability'] = df['danceability'].astype(float)
    return df

# Example usage
if __name__ == "__main__":
    # Load the collected data
    df = pd.read_csv('path_to_your_data.csv')
    processed_df = clean_and_process_data(df)
    print(processed_df)
