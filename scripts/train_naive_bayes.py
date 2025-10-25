import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

def train_baseline_sentiment_model(data_path='data/reviews_processed.csv', grid_search=True, nb__alpha=0.1, tfidf__max_df=0.75, tfidf__ngram_range=(1, 2), sample_size: int = 50000):
    """
    Trains and evaluates a Multinomial Naive Bayes model for sentiment analysis.
    Can optionally perform a grid search.

    Args:
        data_path (str): Path to the processed reviews CSV file.
        grid_search (bool): If True, performs a grid search.
        nb__alpha (float): Alpha for MultinomialNB.
        tfidf__max_df (float): max_df for TfidfVectorizer.
        tfidf__ngram_range (tuple): ngram_range for TfidfVectorizer.
        sample_size (int, optional): Number of reviews to use. If None, uses all.
    """
    # --- 1. Load Data ---
    print(f"Loading data from '{data_path}'...")
    if not os.path.exists(data_path):
        print(f"\nERROR: '{data_path}' not found. Please run the EDA script first!")
        return
        
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    # --- 2. Sample Data ---
    if sample_size:
        print(f"Using a sample of {sample_size} reviews for training the baseline model.")
        df = df.sample(n=sample_size, random_state=42)

    # --- 3. Train-Test Split ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['full_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )

    # --- 4. Create a Pipeline ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB()),
    ])

    best_params = None

    if grid_search:
        # --- 5a. Perform Grid Search ---
        print("Performing Grid Search to find the best hyperparameters...")
        parameters = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.5, 0.75, 1.0],
            'nb__alpha': [0.1, 0.5, 1.0],
        }
        param_grid = list(ParameterGrid(parameters))
        best_score = -1

        for params in tqdm(param_grid, desc="Grid Search Progress"):
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_params = params
        
        print(f"\nBest score on test set: {best_score:.4f}")
        print("Best parameters found:")
        print(best_params)

    else:
        # --- 5b. Use provided hyperparameters ---
        print("Skipping grid search and using provided hyperparameters...")
        best_params = {
            'nb__alpha': nb__alpha,
            'tfidf__max_df': tfidf__max_df,
            'tfidf__ngram_range': tfidf__ngram_range
        }

    # --- 6. Train the Final Model ---
    print("\nTraining final model...")
    best_model = pipeline.set_params(**best_params)
    best_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 7. Evaluate the Best Model ---
    print("\n--- Model Evaluation ---")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    target_names = ['Negative', 'Positive']
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix for Naive Bayes on Amazon Reviews')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
if __name__ == "__main__":
    train_baseline_sentiment_model(sample_size=150000, grid_search=False)