import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from typing import Dict, Tuple, List

class MLModels:
    def __init__(self):
        self.reset_models()

    def reset_models(self):
        """Initialize or reset all models and scalers"""
        self.yield_predictor = LinearRegression()
        self.defect_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.defect_scaler = StandardScaler()
        self.is_yield_fitted = False
        self.is_defect_fitted = False
        self.last_training_features = None
        self.last_training_target = None
        self.feature_names = {
            'yield': ['temperature', 'humidity', 'machine_downtime'],
            'defect': ['temperature', 'humidity', 'production_yield', 'machine_downtime']
        }

    def prepare_yield_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for yield prediction"""
        X = data[self.feature_names['yield']].copy()
        y = data['production_yield'].values.reshape(-1, 1)

        # Save the last training data for future scaling
        self.last_training_features = X.copy()
        self.last_training_target = y.copy()

        # Scale features and target
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)

        return X_scaled, y_scaled.ravel()

    def calculate_feature_importance_yield(self) -> np.ndarray:
        """Calculate feature importance for yield prediction"""
        # For linear regression, use absolute coefficients as importance
        importance = np.abs(self.yield_predictor.coef_)
        # Normalize to sum to 1
        return importance / np.sum(importance)

    def train_yield_predictor(self, data: pd.DataFrame) -> Dict:
        """Train yield prediction model with enhanced evaluation"""
        try:
            print("Starting yield predictor training...")

            X, y = self.prepare_yield_data(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Perform cross-validation
            cv_scores = cross_val_score(
                self.yield_predictor, X, y,
                cv=5, scoring='r2'
            )

            # Fit the model
            self.yield_predictor.fit(X_train, y_train)
            self.is_yield_fitted = True
            print("Model training completed successfully")

            # Get predictions in original scale
            train_pred = self.target_scaler.inverse_transform(
                self.yield_predictor.predict(X_train).reshape(-1, 1)
            ).ravel()
            test_pred = self.target_scaler.inverse_transform(
                self.yield_predictor.predict(X_test).reshape(-1, 1)
            ).ravel()

            y_train_orig = self.target_scaler.inverse_transform(
                y_train.reshape(-1, 1)
            ).ravel()
            y_test_orig = self.target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            metrics = {
                'r2_train': r2_score(y_train_orig, train_pred),
                'r2_test': r2_score(y_test_orig, test_pred),
                'mae_train': mean_absolute_error(y_train_orig, train_pred),
                'mae_test': mean_absolute_error(y_test_orig, test_pred),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'feature_importance': self.calculate_feature_importance_yield().tolist()
            }

            print(f"Training metrics: {metrics}")
            return metrics

        except Exception as e:
            print(f"Error in training yield predictor: {str(e)}")
            self.is_yield_fitted = False
            raise

    def predict_yield(self, temperature: float, humidity: float, downtime: float) -> float:
        """Make yield prediction with validation"""
        try:
            if not self.is_yield_fitted or self.yield_predictor is None:
                raise ValueError("Model not trained yet. Please train the model first.")

            features = np.array([[temperature, humidity, downtime]])

            # Validate input ranges
            if not (20 <= temperature <= 30):
                raise ValueError("Temperature must be between 20°C and 30°C")
            if not (30 <= humidity <= 50):
                raise ValueError("Humidity must be between 30% and 50%")
            if not (0 <= downtime <= 24):
                raise ValueError("Downtime must be between 0 and 24 hours")

            # Scale using the same scaler fitted during training
            if self.last_training_features is not None:
                temp_df = pd.DataFrame(
                    features,
                    columns=self.feature_names['yield']
                )
                features_scaled = self.feature_scaler.transform(temp_df)
            else:
                raise ValueError("Model needs to be retrained. Please train the model first.")

            prediction_scaled = self.yield_predictor.predict(features_scaled)
            prediction = self.target_scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            )[0][0]

            return prediction

        except Exception as e:
            print(f"Error in predicting yield: {str(e)}")
            raise

    def explain_yield_prediction(self, temperature: float, humidity: float,
                               downtime: float) -> Dict[str, str]:
        """Provide explanation for yield prediction"""
        if not self.is_yield_fitted:
            raise ValueError("Model not trained yet")

        # Get feature importance
        importance = self.calculate_feature_importance_yield()
        features = np.array([[temperature, humidity, downtime]])

        # Scale features
        features_scaled = self.feature_scaler.transform(features)

        # Calculate impact
        impacts = features_scaled[0] * self.yield_predictor.coef_

        # Create explanation
        explanation = {}
        for feat, imp, impact in zip(self.feature_names['yield'], importance, impacts):
            direction = "increasing" if impact > 0 else "decreasing"
            strength = "strong" if abs(imp) > 0.4 else (
                "moderate" if abs(imp) > 0.2 else "weak"
            )
            explanation[feat.title()] = f"{strength} impact ({direction} yield)"

        return explanation

    def prepare_defect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for defect classification"""
        X = data[self.feature_names['defect']].copy()

        # Create binary classification target (high/low defect rate)
        defect_threshold = data['defect_rate'].median()
        y = (data['defect_rate'] > defect_threshold).astype(int)

        X_scaled = self.defect_scaler.fit_transform(X)
        return X_scaled, y

    def train_defect_classifier(self, data: pd.DataFrame) -> Dict:
        """Train defect classification model with enhanced evaluation"""
        try:
            print("Starting defect classifier training...")

            X, y = self.prepare_defect_data(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Perform cross-validation
            cv_scores = cross_val_score(
                self.defect_classifier, X, y,
                cv=5, scoring='accuracy'
            )

            self.defect_classifier.fit(X_train, y_train)
            self.is_defect_fitted = True

            # Make predictions
            train_pred = self.defect_classifier.predict(X_train)
            test_pred = self.defect_classifier.predict(X_test)

            # Calculate metrics
            metrics = {
                'accuracy_train': accuracy_score(y_train, train_pred),
                'accuracy_test': accuracy_score(y_test, test_pred),
                'precision': precision_score(y_test, test_pred),
                'recall': recall_score(y_test, test_pred),
                'f1': f1_score(y_test, test_pred),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'feature_importance': self.defect_classifier.feature_importances_.tolist(),
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
            }

            print(f"Training metrics: {metrics}")
            return metrics

        except Exception as e:
            print(f"Error in training defect classifier: {str(e)}")
            self.is_defect_fitted = False
            raise

    def predict_defect_risk_proba(self, temperature: float, humidity: float,
                                yield_val: float, downtime: float) -> float:
        """Predict defect risk probability"""
        try:
            if not self.is_defect_fitted:
                raise ValueError("Model not trained yet. Please train the model first.")

            # Validate input ranges
            if not (20 <= temperature <= 30):
                raise ValueError("Temperature must be between 20°C and 30°C")
            if not (30 <= humidity <= 50):
                raise ValueError("Humidity must be between 30% and 50%")
            if not (70 <= yield_val <= 100):
                raise ValueError("Production yield must be between 70% and 100%")
            if not (0 <= downtime <= 24):
                raise ValueError("Downtime must be between 0 and 24 hours")

            features = np.array([[temperature, humidity, yield_val, downtime]])
            features_scaled = self.defect_scaler.transform(features)

            # Get probability of high defect risk
            probabilities = self.defect_classifier.predict_proba(features_scaled)[0]
            return probabilities[1]  # Return probability of high risk

        except Exception as e:
            print(f"Error in predicting defect risk: {str(e)}")
            raise

    def explain_defect_prediction(self, temperature: float, humidity: float,
                                yield_val: float, downtime: float) -> Dict[str, str]:
        """Provide explanation for defect risk prediction"""
        if not self.is_defect_fitted:
            raise ValueError("Model not trained yet")

        # Get feature importance
        importance = self.defect_classifier.feature_importances_
        features = np.array([[temperature, humidity, yield_val, downtime]])

        # Scale features
        features_scaled = self.defect_scaler.transform(features)

        # Create explanation
        explanation = {}
        feature_ranges = {
            'Temperature': (20, 30),
            'Humidity': (30, 50),
            'Production Yield': (70, 100),
            'Downtime': (0, 24)
        }

        for feat, imp, val, (min_val, max_val) in zip(
            self.feature_names['defect'],
            importance,
            features[0],
            feature_ranges.values()
        ):
            # Calculate where the value falls in its range
            range_position = (val - min_val) / (max_val - min_val)
            if range_position < 0.33:
                level = "low"
            elif range_position < 0.66:
                level = "moderate"
            else:
                level = "high"

            importance_level = "strong" if imp > 0.3 else (
                "moderate" if imp > 0.15 else "weak"
            )

            explanation[feat.title()] = f"{importance_level} factor ({level} value)"

        return explanation