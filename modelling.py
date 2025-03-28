# modeling.py

import re
import string
import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Import the optimized cleaning function from preprocessing.py
from preprocessing import clean_text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the description column.
    """
    df = df.copy()
    df['clean_description'] = df['description'].apply(clean_text)
    return df

def get_models():
    """
    Define multiple models to train.
    Returns a dictionary with model names as keys and sklearn pipelines as values.
    """
    pipelines = dict()

    # Pipeline: Tfidf vectorizer + Multinomial Naive Bayes
    pipelines['MultinomialNB'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])

    # Pipeline: Tfidf vectorizer + Logistic Regression
    pipelines['LogisticRegression'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Pipeline: Tfidf vectorizer + Linear SVC
    pipelines['LinearSVC'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC())
    ])

    # Pipeline: Tfidf vectorizer + K-Nearest Neighbors
    pipelines['KNN'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', KNeighborsClassifier())
    ])

    # Pipeline: Tfidf vectorizer + Random Forest
    pipelines['RandomForest'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Pipeline: Tfidf vectorizer + Bernoulli Naive Bayes
    pipelines['BernoulliNB'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', BernoulliNB())
    ])

    # Pipeline: Tfidf vectorizer + Neural Network (MLPClassifier)
    pipelines['MLPClassifier'] = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42))
    ])

    return pipelines

def analyze_feature_importance(pipeline: Pipeline):
    """
    If the classifier supports feature importance, display the top 10 features.
    Works for classifiers with `feature_importances_` or `coef_` attributes.
    """
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()

    # Check for feature_importances_ attribute (e.g., RandomForestClassifier)
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print("\nTop 10 Features (by importance):")
        for idx in indices:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")

    # Check for coef_ attribute (e.g., LogisticRegression, LinearSVC)
    elif hasattr(classifier, "coef_"):
        # For multi-class, sum the absolute weights across classes
        coefs = classifier.coef_
        if coefs.ndim == 2:
            importance = np.mean(np.abs(coefs), axis=0)
        else:
            importance = np.abs(coefs)
        indices = np.argsort(importance)[::-1][:10]
        print("\nTop 10 Features (by coefficient magnitude):")
        for idx in indices:
            print(f"{feature_names[idx]}: {importance[idx]:.4f}")
    else:
        print("\nFeature importance is not available for this classifier.")

def analyze_misclassifications(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, n_examples: int = 10):
    """
    Print sample misclassified examples with their true and predicted labels.
    """
    predictions = pipeline.predict(X_test)
    misclassified = X_test[predictions != y_test]
    true_labels = y_test[predictions != y_test]
    predicted_labels = predictions[predictions != y_test]

    print(f"\nMisclassified Examples (first {n_examples}):")
    for i, (text, true_label, pred_label) in enumerate(zip(misclassified, true_labels, predicted_labels)):
        if i >= n_examples:
            break
        print(f"\nExample {i+1}:")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label}")
        print(f"Text: {text[:300]}{'...' if len(text)>300 else ''}")

def train_and_evaluate(df: pd.DataFrame, models: dict):
    """
    Split the data, train each model, and print out performance metrics.
    Also provides feature importance and misclassification analysis for the best model.
    """
    X = df['clean_description']
    y = df['genre']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    model_pipelines = {}
    for model_name, pipeline in models.items():
        print(f"\nTraining model: {model_name}")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = accuracy
        model_pipelines[model_name] = pipeline
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))
    
    # Determine the best model
    best_model_name = max(results, key=results.get)
    best_pipeline = model_pipelines[best_model_name]
    print(f"\nBest performing model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

    # Analyze feature importance for the best model (if available)
    analyze_feature_importance(best_pipeline)

    # Analyze misclassifications for the best model
    analyze_misclassifications(best_pipeline, X_test, y_test)

def main():
    # Update path as needed
    train_path = "./Genre Classification Dataset/train_data.txt"
    df = pd.read_csv(train_path, sep=':::', names=['title', 'genre', 'description'], engine="python")
    
    # Preprocess data using our modular cleaning function
    df = preprocess_data(df)
    
    # Get models and train/evaluate
    models = get_models()
    train_and_evaluate(df, models)

if __name__ == "__main__":
    main()
