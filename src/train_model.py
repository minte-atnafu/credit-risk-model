import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name, experiment_id):
    """Train model with GridSearchCV and log to MLflow."""
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_GridSearch") as run:
        try:
            logging.info(f"Training {model_name}...")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, error_score='raise')
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            logging.info(f"{model_name} metrics: {metrics}")
            
            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            
            return best_model, metrics, grid_search.best_params_, run.info.run_id
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            raise

def main(input_path, experiment_name="CreditRiskExperiment"):
    """Main function for model training and tracking."""
    # Load data
    logging.info(f"Loading data from {input_path}...")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Data file {input_path} does not exist.")
    df = pd.read_csv(input_path)
    
    # Verify required columns
    if 'is_high_risk' not in df.columns:
        logging.error(f"is_high_risk missing in dataset. Available columns: {df.columns.tolist()}")
        raise ValueError("is_high_risk column not found.")
    
    # Define non-numeric columns to drop
    non_numeric_columns = [
        'remainder__TransactionId', 'remainder__BatchId', 'remainder__AccountId',
        'remainder__SubscriptionId', 'remainder__CustomerId', 'remainder__TransactionStartTime'
    ]
    drop_columns = ['is_high_risk', 'FraudResult'] + [col for col in non_numeric_columns if col in df.columns]
    
    # Prepare features and target
    X = df.drop(columns=drop_columns)
    y = df['is_high_risk']
    
    # Log feature columns
    logging.info(f"Feature columns: {X.columns.tolist()}")
    
    # Split data
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # Define models and parameter grids
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'param_grid': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            }
        }
    }
    
    # Train and evaluate models
    best_roc_auc = 0
    best_model_name = None
    best_model_run_id = None
    
    for model_name, config in models.items():
        try:
            model, metrics, best_params, run_id = train_and_evaluate_model(
                config['model'], config['param_grid'], X_train, X_test, y_train, y_test,
                model_name, experiment.experiment_id
            )
            
            # Track best model
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_model_name = model_name
                best_model_run_id = run_id
        except Exception as e:
            logging.error(f"Failed to train {model_name}: {str(e)}")
            continue
    
    # Register best model
    if best_model_run_id:
        logging.info(f"Registering best model: {best_model_name} with ROC-AUC: {best_roc_auc}")
        model_uri = f"runs:/{best_model_run_id}/{best_model_name}_model"
        mlflow.register_model(model_uri, "CreditRiskModel")
    else:
        logging.error("No models were successfully trained.")
        raise ValueError("No models were successfully trained.")
    
    logging.info("Model training and registration completed.")

if __name__ == "__main__":
    input_path = '../data/processed/transformed_transactions_with_risk.csv'
    try:
        main(input_path)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error: {str(e)}")