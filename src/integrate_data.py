import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The CSV file is empty.")
        print(f"Data loaded successfully from {file_path} with {len(df)} records.")
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

def main():
    # Load the existing dataset
    existing_df = load_data('music_data.csv')

    # Load the new dataset
    new_df = load_data('new_music_data.csv')

    if existing_df is not None and new_df is not None:
        # Combine the datasets
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates().reset_index(drop=True)
        
        # Save the combined dataset
        combined_df.to_csv('combined_music_data.csv', index=False)
        print(f"Combined dataset saved with {len(combined_df)} records.")

if __name__ == "__main__":
    main()
