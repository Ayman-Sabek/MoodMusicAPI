import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import joblib

def load_data(file_path):
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
    print(f"Data before dropping missing values: {len(df)} records.")
    missing_values = df.isnull().sum()
    print(f"Missing values in each column:\n{missing_values}")

    print(f"Data after checking missing values in essential columns: {len(df)} records.")

    df['energy'] = df['energy'].astype(float)
    df['valence'] = df['valence'].astype(float)
    df['tempo'] = df['tempo'].astype(float)
    df['danceability'] = df['danceability'].astype(float)

    df['energy'] = (df['energy'] - df['energy'].min()) / (df['energy'].max() - df['energy'].min())
    df['valence'] = (df['valence'] - df['valence'].min()) / (df['valence'].max() - df['valence'].min())
    df['tempo'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
    df['danceability'] = (df['danceability'] - df['danceability'].min()) / (df['danceability'].max() - df['danceability'].min())

    print("Data processing completed.")
    return df

def feature_engineering(df):
    df['mood_score'] = (df['energy'] + df['valence']) / 2
    df['energy_danceability'] = df['energy'] * df['danceability']
    df['tempo'] = df['tempo'].apply(lambda x: x if x > 1e-6 else 1e-6)
    print(f"Number of zero values in 'tempo' before log transformation: {(df['tempo'] == 0).sum()}")
    df['log_tempo'] = df['tempo'].apply(lambda x: np.log(x))
    print("Feature engineering completed.")
    return df

def handle_infinite_values(df):
    print("Handling infinite values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print(f"Data after handling infinite values:\n{df.describe()}")
    return df

def tune_and_evaluate_model(X, y, model, param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X, y)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {model.__class__.__name__}: {grid_search.best_score_}")
    return grid_search.best_estimator_

load_dotenv()

file_path = 'extended_music_data.csv'
df = load_data(file_path)
if df is not None:
    processed_df = clean_and_process_data(df)
    if processed_df is not None and not processed_df.empty:
        engineered_df = feature_engineering(processed_df)
        print(f"Data after feature engineering:\n{engineered_df.describe()}")
        engineered_df = handle_infinite_values(engineered_df)
        if not np.isfinite(engineered_df.select_dtypes(include=[np.number])).all().all():
            print("There are still infinite values in the dataset.")
        else:
            print("No infinite values remain in the dataset.")

        X = engineered_df[['energy', 'valence', 'tempo', 'danceability', 'mood_score', 'log_tempo', 'energy_danceability']]
        y = engineered_df['key']

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        class_distribution = y_res.value_counts()
        print(f"Class distribution after resampling:\n{class_distribution}")

        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }

        param_grids = {
            "Random Forest": {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30]
            },
            "Support Vector Machine": {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf', 'poly']
            },
            "K-Nearest Neighbors": {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            }
        }

        best_models = {}

        for name, model in models.items():
            print(f"Evaluating {name}")
            best_model = tune_and_evaluate_model(X_res, y_res, model, param_grids[name])
            best_models[name] = best_model  # Save the best model in the dictionary

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print(f"Final Evaluation with {name}:")
            print("Accuracy:", accuracy)
            print("Classification Report:\n", report)

        # Save the best models
        joblib.dump(best_models["Random Forest"], 'best_random_forest_model.pkl')
        joblib.dump(best_models["Support Vector Machine"], 'best_svm_model.pkl')
        joblib.dump(best_models["K-Nearest Neighbors"], 'best_knn_model.pkl')
    else:
        print("No data available after cleaning.")
else:
    print("Data loading failed.")
