"""
Credit Card Fraud Detection System
--------------------------------
This module implements a stacking-based machine learning system for detecting credit card fraud.
It uses a combination of Random Forest, XGBoost, and Gradient Boosting classifiers with
Logistic Regression as a meta-classifier.

Key Features:
- Incremental data processing for large datasets
- Advanced feature engineering with PCA
- Class imbalance handling using SMOTE-ENN
- Comprehensive model evaluation and visualization
- Memory-efficient chunk processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,precision_score,
    f1_score,precision_recall_curve, roc_curve, auc, average_precision_score
)
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier   

class FraudDetectionSystem:
    """
    A comprehensive system for credit card fraud detection using stacking ensemble.
    """
    
    def __init__(self, features, chunk_size=10000):
        """
        Initialize the fraud detection system.
        
        Args:
            features (list): List of feature names to use for prediction
            chunk_size (int): Size of chunks for processing large datasets
        """
        self.features = features
        self.chunk_size = chunk_size
        self.scaler = StandardScaler()
        self.ipca = IncrementalPCA(n_components=3)
        self.model = None
        
    def load_data_in_chunks(self, file_path):
        """
        Generator function to load data in chunks.
        
        Args:
            file_path (str): Path to the CSV file
            
        Yields:
            pandas.DataFrame: Chunk of data
        """
        return pd.read_csv(file_path, chunksize=self.chunk_size)

    def process_chunk(self, chunk):
        """
        Process a single chunk of data.
        
        Args:
            chunk (pandas.DataFrame): Data chunk to process
            
        Returns:
            numpy.ndarray: Processed features
        """
        X = chunk[self.features].fillna(chunk[self.features].mean())
        X_scaled = self.scaler.transform(X)
        X_pca = self.ipca.transform(X_scaled)
        return X_pca

    def process_data(self, file_path):
        """
        Process the entire dataset in chunks.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            tuple: (X_all, y_all) - Processed features and labels
        """
        # First pass: Fit scaler and IncrementalPCA
        for chunk in self.load_data_in_chunks(file_path):
            X = chunk[self.features].fillna(chunk[self.features].mean())
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
            self.ipca.partial_fit(X_scaled)

        # Second pass: Transform data
        X_all, y_all = [], []
        for chunk in self.load_data_in_chunks(file_path):
            X_pca = self.process_chunk(chunk)
            X_all.append(X_pca)
            y_all.extend(chunk['is_fraud'])

        return np.vstack(X_all), np.array(y_all)

    def handle_imbalance(self, X, y):
        """
        Apply SMOTE-ENN to handle class imbalance.
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
            
        Returns:
            tuple: (X_resampled, y_resampled) - Balanced dataset
        """
        smote_enn = SMOTEENN(random_state=42)
        return smote_enn.fit_resample(X, y)

    def build_model(self):
        """
        Build the stacking classifier model.
        
        Returns:
            StackingClassifier: Configured stacking ensemble model
        """
        base_models = [
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ]
        meta_clf = LogisticRegression(random_state=42)
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_clf,
            cv=5
        )

    def train(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
        """
        self.model = self.build_model()
        self.model.fit(X, y)

    def evaluate(self, X, y_true):
        """
        Evaluate the model and return performance metrics.
        
        Args:
            X (numpy.ndarray): Features
            y_true (numpy.ndarray): True labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': precision_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),  # Add f1_score
            'roc_auc': roc_auc_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }

    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path where to save the model
        """
        if self.model is not None:
            joblib.dump(self.model, filepath)
        else:
            raise ValueError("No model to save. Please train the model first.")

    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = joblib.load(filepath)
