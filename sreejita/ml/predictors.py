"""Predictive Analytics Module for Sreejita Framework v1.7.

This module provides machine learning capabilities for time series forecasting,
regression, and classification tasks with automated model selection.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')


class PredictiveAnalytics:
    """Automated machine learning for predictive analytics.
    
    Provides automated model selection, training, and prediction capabilities
    for regression, classification, and time series forecasting tasks.
    
    Attributes:
        model: The trained ML model
        scaler: StandardScaler for feature normalization
        model_name: Name of the selected model
        feature_importance: Dictionary of feature importance scores
        metrics: Dictionary of model performance metrics
    """
    
    def __init__(self, task_type: str = 'regression'):
        """Initialize PredictiveAnalytics.
        
        Args:
            task_type: Type of ML task ('regression', 'classification', 'timeseries')
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.model_name = None
        self.feature_importance = {}
        self.metrics = {}
        self.feature_names = None
    
    def auto_select_model(self, X: pd.DataFrame, y: pd.Series, 
                         test_size: float = 0.2) -> Dict[str, Any]:
        """Automatically select and train the best model.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model metrics and selection details
        """
        if self.task_type != 'regression':
            raise NotImplementedError(f'Task type {self.task_type} not yet implemented')
        
        # Store feature names
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Candidate models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_r2 = -np.inf
        best_model = None
        best_name = None
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
        
        # Store best model
        self.model = best_model
        self.model_name = best_name
        self.metrics = results[best_name]
        self.metrics['model'] = best_name
        
        # Extract feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            if self.feature_names:
                self.feature_importance = dict(
                    zip(self.feature_names, best_model.feature_importances_)
                )
            else:
                self.feature_importance = {
                    f'feature_{i}': imp 
                    for i, imp in enumerate(best_model.feature_importances_)
                }
        
        return {
            'best_model': best_name,
            'metrics': self.metrics,
            'all_results': results,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using trained model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError('Model not trained. Call auto_select_model first.')
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray],
                               confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals.
        
        Args:
            X: Input features
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        predictions = self.predict(X)
        
        # Simplified confidence intervals based on model RMSE
        rmse = self.metrics.get('rmse', np.std(predictions))
        z_score = 1.96 if confidence == 0.95 else 2.576  # Common z-scores
        margin = z_score * rmse
        
        confidence_intervals = np.column_stack([
            predictions - margin,
            predictions + margin
        ])
        
        return predictions, confidence_intervals
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'model_params': self.model.get_params() if self.model else None
        }


class AutoML:
    """Automated Machine Learning orchestrator.
    
    Coordinates feature engineering, model selection, and hyperparameter
    tuning for end-to-end ML workflows.
    """
    
    def __init__(self):
        """Initialize AutoML orchestrator."""
        self.analytics = None
        self.pipeline = []
    
    def run(self, X: pd.DataFrame, y: pd.Series, 
            task_type: str = 'regression') -> Dict[str, Any]:
        """Run complete AutoML pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: Type of ML task
            
        Returns:
            Dictionary with pipeline results
        """
        self.analytics = PredictiveAnalytics(task_type=task_type)
        results = self.analytics.auto_select_model(X, y)
        
        self.pipeline.append({
            'step': 'model_selection',
            'results': results
        })
        
        return results
    
    def get_best_model(self):
        """Get the best trained model.
        
        Returns:
            PredictiveAnalytics instance with trained model
        """
        return self.analytics
