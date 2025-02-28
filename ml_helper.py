import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px

class AutoML:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        self.model = None
        self.target = None
        self.features = None
        self.task_type = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_data(self, target_column, features=None):
        """Prepare data for modeling"""
        if features is None:
            features = [col for col in self.numeric_cols if col != target_column]
            
        self.target = target_column
        self.features = features
        
        # Determine task type
        if self.df[target_column].dtype in ['int64', 'float64']:
            if len(self.df[target_column].unique()) < 10:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        else:
            self.task_type = 'classification'
            
        # Prepare feature matrix
        X = self.df[features].copy()
        y = self.df[target_column].copy()
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Handle categorical target for classification
        if self.task_type == 'classification' and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
        return X, y
    
    def train_model(self, target_column, features=None):
        """Train the best model for the data"""
        X, y = self.prepare_data(target_column, features)
        
        if self.task_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42)
            }
            scoring = 'neg_mean_squared_error'
        else:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42)
            }
            scoring = 'accuracy'
            
        # Find best model
        best_score = float('-inf')
        best_model = None
        model_scores = {}
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            avg_score = scores.mean()
            model_scores[name] = avg_score
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train final model
        self.model = best_model.fit(X, y)
        return model_scores
    
    def get_feature_importance(self):
        """Get feature importance or coefficients"""
        if self.model is None:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return None
            
        return dict(zip(self.features, importance))
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            return None
            
        # Prepare input data
        X = input_data[self.features].copy()
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']):
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale features
        X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            return None
            
        y_pred = self.model.predict(X_test)
        
        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.model.score(X_test, y_test)
            
            return {
                'RMSE': rmse,
                'R2 Score': r2
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            return {
                'Accuracy': accuracy,
                'Classification Report': report
            }
