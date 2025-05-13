import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
import joblib

# Set up logging with proper configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

class StockDirectionModel:
    """Class to handle stock market direction prediction model training and evaluation."""
    
    def __init__(self, data_path="data/processed/combined_features.csv", test_size=0.2):
        """Initialize the model with data path and test size.
        
        Args:
            data_path (str): Path to the processed data
            test_size (float): Proportion of data to use for testing
        """
        self.data_path = data_path
        self.test_size = test_size
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare data for modeling.
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Ensure the file exists
            if not os.path.exists(self.data_path):
                logger.error(f"Data file not found: {self.data_path}")
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
                
            # Load the data
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            logger.info(f"Columns in DataFrame: {df.columns.tolist()}")
            
            # Create target variable if it doesn't exist
            if 'Target' not in df.columns:
                logger.info("Creating target variable based on next day returns")
                # Use Next_Day_Return to create a binary target
                if 'Next_Day_Return' in df.columns:
                    df['Target'] = np.where(df['Next_Day_Return'] > 0, 1, 0)
                    logger.info("Created target variable from Next_Day_Return")
                # Fallback to Price_Change if Next_Day_Return doesn't exist
                elif 'Price_Change' in df.columns:
                    df['Target'] = np.where(df['Price_Change'].shift(-1) > 0, 1, 0)
                    logger.info("Created target variable from shifted Price_Change")
                else:
                    logger.error("Cannot create target variable: required columns not found")
                    raise ValueError("Cannot create target variable: Next_Day_Return or Price_Change required")
            
            # Drop NaN values
            initial_rows = len(df)
            df.dropna(inplace=True)
            final_rows = len(df)
            logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values")
            
            # Select feature columns
            # Start with basic technical indicators
            potential_features = [
                # Momentum indicators
                'RSI_14', 'Stoch_K', 'Stoch_D', 'ROC_10', 'Williams_R',
                # Trend indicators
                'SMA_10', 'SMA_50', 'EMA_20', 'MACD', 'MACD_Signal', 'ADX',
                # Volatility indicators
                'BB_Width_20', 'ATR_14', 'Volatility_10', 'Volatility_20',
                # Volume indicators
                'OBV', 'CMF_10', 'Volume_Ratio_10',
                # Price indicators
                'Price_Change', 'Price_Change_5d', 'Log_Return',
                # Combined signals
                'Bull_Signal', 'Bear_Signal', 'Strong_Trend', 'MA_Convergence'
            ]
            
            # Add sentiment score if available
            if 'sentiment_score' in df.columns:
                potential_features.append('sentiment_score')
                
            # Check which features are available
            available_features = [col for col in potential_features if col in df.columns]
            logger.info(f"Selected {len(available_features)} features from {len(potential_features)} potential features")
            
            if not available_features:
                logger.error("No usable features found in the dataset")
                raise ValueError("No usable features found in the dataset")
                
            # Store the features for later use
            self.feature_columns = available_features
            logger.info(f"Using features: {self.feature_columns}")
            
            # Prepare X and y
            X = df[self.feature_columns]
            y = df['Target']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False, random_state=42
            )
            
            # Apply feature scaling
            self.X_train = self.scaler.fit_transform(X_train)
            self.X_test = self.scaler.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test
            
            logger.info(f"Data split: X_train={self.X_train.shape}, X_test={self.X_test.shape}")
            
            # Check for class imbalance
            train_class_distribution = y_train.value_counts(normalize=True)
            logger.info(f"Class distribution in training set: {train_class_distribution.to_dict()}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_model(self, hyperparameter_tuning=False):
        """Train the XGBoost model.
        
        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            
        Returns:
            xgboost.XGBClassifier: Trained model
        """
        try:
            if self.X_train is None or self.y_train is None:
                logger.info("Data not loaded. Loading data first...")
                self.load_data()
                
            logger.info("Training XGBoost model...")
            
            if hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning with TimeSeriesSplit...")
                
                # Define parameter grid
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
                
                # Initialize model
                base_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )
                
                # Initialize time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Initialize grid search
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='roc_auc',
                    verbose=1,
                    n_jobs=-1
                )
                
                # Fit grid search
                grid_search.fit(self.X_train, self.y_train)
                
                # Get best model
                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best ROC AUC score: {grid_search.best_score_:.4f}")
                
            else:
                # Initialize model with default parameters
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=3,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )
                
                # Create evaluation set for early stopping
                eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
                
                # Fit model
                self.model.fit(
                    self.X_train, 
                    self.y_train,
                    eval_set=eval_set,
                    verbose=True
                )
                
                logger.info(f"Model trained with {self.model.n_estimators} trees")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(self):
        """Evaluate the trained model.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("Model has not been trained yet")
                raise ValueError("Model has not been trained yet")
                
            logger.info("Evaluating model performance...")
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Log metrics
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            logger.info(f"Classification Report:\n{pd.DataFrame(class_report).transpose()}")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': self.model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                logger.info(f"Top 10 important features:\n{feature_importance.head(10)}")
                
            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'roc_auc': roc_auc,
                'feature_importance': feature_importance if hasattr(self.model, 'feature_importances_') else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model_dir="models"):
        """Save the trained model to disk.
        
        Args:
            model_dir (str): Directory to save the model
        """
        try:
            if self.model is None:
                logger.error("No trained model to save")
                raise ValueError("No trained model to save")
                
            # Ensure directory exists
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            
            # Create model filename with timestamp
            model_path = os.path.join(model_dir, "xgboost_stock_direction_model.joblib")
            scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
            feature_list_path = os.path.join(model_dir, "feature_list.txt")
            
            # Save model and scaler
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature list
            with open(feature_list_path, 'w') as f:
                f.write('\n'.join(self.feature_columns))
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Feature list saved to {feature_list_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def plot_results(self, output_dir="results"):
        """Plot evaluation results and save figures.
        
        Args:
            output_dir (str): Directory to save the figures
        """
        try:
            if self.model is None:
                logger.error("Model has not been trained yet")
                raise ValueError("Model has not been trained yet")
                
            # Ensure directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Down', 'Up'], 
                        yticklabels=['Down', 'Up'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = precision_recall_curve(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
            plt.close()
            
            # Plot feature importance
            if hasattr(self.model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': self.model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                # Plot top 15 features
                top_features = feature_importance.head(15)
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()
                
            logger.info(f"Evaluation plots saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise


def main():
    """Main function to run the model training and evaluation."""
    try:
        # Initialize and train the model
        stock_model = StockDirectionModel()
        
        # Load data
        stock_model.load_data()
        
        # Train model
        logger.info("Starting model training...")
        stock_model.train_model(hyperparameter_tuning=False)
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_results = stock_model.evaluate_model()
        
        # Save model
        stock_model.save_model()
        
        # Plot results
        stock_model.plot_results()
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()