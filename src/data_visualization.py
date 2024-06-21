import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button
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

def plot_mood_score(df):
    plt.clf()
    sns.histplot(df['mood_score'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Mood Score')
    plt.xlabel('Mood Score')
    plt.ylabel('Frequency')
    plt.draw()

def plot_energy_by_language(df):
    plt.clf()
    sns.boxplot(x='language', y='energy', data=df)
    plt.title('Distribution of Energy by Language')
    plt.xlabel('Language')
    plt.ylabel('Energy')
    plt.xticks(rotation=45)
    plt.draw()

def plot_energy_vs_danceability(df):
    plt.clf()
    sns.scatterplot(x='energy', y='danceability', hue='language', data=df)
    plt.title('Energy vs. Danceability')
    plt.xlabel('Energy')
    plt.ylabel('Danceability')
    plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.draw()

def plot_tempo_by_language(df):
    plt.clf()
    sns.boxplot(x='language', y='tempo', data=df)
    plt.title('Box Plot of Tempo by Language')
    plt.xlabel('Language')
    plt.ylabel('Tempo')
    plt.xticks(rotation=45)
    plt.draw()

def plot_correlation_heatmap(df):
    plt.clf()
    correlation_matrix = df[['energy', 'valence', 'tempo', 'danceability', 'mood_score']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.draw()

class PlotNavigator:
    def __init__(self, df):
        self.df = df
        self.plots = [plot_mood_score, plot_energy_by_language, plot_energy_vs_danceability, plot_tempo_by_language, plot_correlation_heatmap]
        self.index = 0
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.plot_current()

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_plot)

        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_plot)

    def plot_current(self):
        self.plots[self.index](self.df)
        self.fig.canvas.draw()

    def next_plot(self, event):
        self.index = (self.index + 1) % len(self.plots)
        self.plot_current()

    def prev_plot(self, event):
        self.index = (self.index - 1) % len(self.plots)
        self.plot_current()

def visualize_data(df):
    navigator = PlotNavigator(df)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    file_path = 'music_data.csv'  # Ensure this is the correct path to your CSV file
    df = load_data(file_path)
    if df is not None:
        processed_df = clean_and_process_data(df)
        engineered_df = feature_engineering(processed_df)
        visualize_data(engineered_df)
