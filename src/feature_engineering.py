import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
import os
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up basic logging to monitor the script's execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Column Definitions ---
# These lists define column groups for different processing steps.
NUMERICAL_COLS = ['Amount', 'Value', 'CountryCode', 'PricingStrategy']
CATEGORICAL_COLS = ['CurrencyCode', 'ProductCategory', 'ChannelId', 'ProviderId', 'ProductId']
REMAINDER_COLS = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
IMPUTE_CATEGORICAL_COLS = ['CurrencyCode', 'ProductCategory', 'ChannelId', 'ProviderId']  # Exclude ProductId from imputation

# --- Expected ProductId Categories ---
# This list is used to validate and clean the 'ProductId' column.
EXPECTED_PRODUCT_IDS = [
    'productid_1', 'productid_2', 'productid_3', 'productid_4', 'productid_5',
    'productid_6', 'productid_7', 'productid_8', 'productid_9', 'productid_10',
    'productid_11', 'productid_12', 'productid_13', 'productid_14', 'productid_15',
    'productid_16', 'productid_19', 'productid_20', 'productid_21', 'productid_22',
    'productid_23', 'productid_24', 'productid_27'
]

# --- Custom Transformers ---

class AggregateFeatures:
    """Custom transformer to compute aggregate transaction features per CustomerId."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data, so fit does nothing.
        return self

    def transform(self, X):
        logging.info("Computing aggregate features...")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if 'CustomerId' not in X.columns or 'Amount' not in X.columns:
            raise ValueError("Required columns 'CustomerId' and 'Amount' not found for aggregation.")
        
        # Group by CustomerId and calculate aggregate statistics
        agg_df = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
        }).reset_index()
        agg_df.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount']
        agg_df['StdAmount'] = agg_df['StdAmount'].fillna(0) # Fill NaN for customers with only one transaction
        
        # Merge the new aggregate features back into the original dataframe
        X = X.merge(agg_df, on='CustomerId', how='left')
        logging.info(f"Columns after AggregateFeatures: {list(X.columns)}")
        return X

class DatetimeFeatures:
    """Custom transformer to extract datetime features from TransactionStartTime."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # This transformer is stateless, so fit does nothing.
        return self

    def transform(self, X):
        logging.info("Extracting datetime features...")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if 'TransactionStartTime' not in X.columns:
            raise ValueError("Required column 'TransactionStartTime' not found for datetime extraction.")
        
        X = X.copy() # Avoid SettingWithCopyWarning
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], errors='coerce')
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        logging.info(f"Columns after DatetimeFeatures: {list(X.columns)}")
        return X

# --- Pipeline Creation ---

def create_feature_engineering_pipeline(product_id_categories):
    """Creates a full feature engineering pipeline for the transaction data."""
    logging.info("Creating feature engineering pipeline...")
    
    # Pipeline for numerical features: impute missing values with the median, then scale.
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for most categorical features: impute missing values, then one-hot encode.
    categorical_pipeline_impute = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Special pipeline for 'ProductId': one-hot encode with a fixed set of categories.
    # 'handle_unknown' is set to 'error' to catch any unexpected values that slip through.
    categorical_pipeline_no_impute = Pipeline([
        ('onehot', OneHotEncoder(categories=[product_id_categories], handle_unknown='error', sparse_output=False))
    ])
    
    # The preprocessor applies the correct pipeline to each column group.
    # 'remainder=passthrough' ensures that columns not specified are kept.
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('cat_impute', categorical_pipeline_impute, IMPUTE_CATEGORICAL_COLS),
        ('cat_no_impute', categorical_pipeline_no_impute, ['ProductId'])
    ], remainder='passthrough')
    
    # The final pipeline chains the custom transformers and the preprocessor.
    pipeline = Pipeline([
        ('aggregate', AggregateFeatures()),
        ('datetime', DatetimeFeatures()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

# --- Main Data Processing Function ---

def process_data(input_path, output_path, target_col='FraudResult'):
    """Loads, processes, and saves the transformed transaction data."""
    logging.info(f"Starting data processing for {input_path}...")
    
    # Validate input file exists
    if not os.path.isfile(input_path):
        logging.error(f"Input file not found at {input_path}")
        raise FileNotFoundError(f"File {input_path} does not exist.")
    
    df = pd.read_csv(input_path)
    logging.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Clean 'ProductId' by stripping whitespace and converting to lowercase
    df['ProductId'] = df['ProductId'].str.strip().str.lower()
    
    # Group any 'ProductId' not in the expected list into a single 'other' category
    unexpected = set(df['ProductId'].unique()) - set(EXPECTED_PRODUCT_IDS)
    if unexpected:
        logging.warning(f"Found {len(unexpected)} unexpected ProductId values. Grouping them into 'other'.")
        df['ProductId'] = df['ProductId'].where(df['ProductId'].isin(EXPECTED_PRODUCT_IDS), 'other')
    
    # Determine the final list of ProductId categories for the encoder
    product_id_categories = sorted(df['ProductId'].unique().tolist())
    
    # Separate features (X) from the target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Create and apply the full feature engineering pipeline
    pipeline = create_feature_engineering_pipeline(product_id_categories)
    
    logging.info("Fitting and transforming data with the pipeline...")
    try:
        X_transformed = pipeline.fit_transform(X)
    except Exception as e:
        logging.error(f"Pipeline transformation failed: {e}", exc_info=True)
        raise
    
    # --- Correctly get feature names from the pipeline ---
    # This is the robust way to get names, preventing mismatch errors.
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    logging.info(f"Transformed data shape: {X_transformed.shape}")
    logging.info(f"Number of features generated: {len(feature_names)}")
    
    # Final validation check for shape mismatch
    if X_transformed.shape[1] != len(feature_names):
        logging.error(f"Shape mismatch POST-processing. Columns: {X_transformed.shape[1]}, Names: {len(feature_names)}")
        raise ValueError("Final shape mismatch detected between transformed data and feature names.")
    
    # Convert the transformed numpy array back to a DataFrame with correct column names
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    # Add the target column back to the transformed dataframe
    X_transformed_df[target_col] = y
    
    # Save the final, processed data to a new CSV file
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    X_transformed_df.to_csv(output_path, index=False)
    logging.info(f"Transformed data successfully saved to {output_path}")
    
    return X_transformed_df

# --- Script Execution Block ---

if __name__ == "__main__":
    # Define file paths
    default_input_path = '../data/raw/data.csv'
    default_output_path = '../data/processed/transformed_transactions.csv'
    
    try:
        # Run the main processing function
        transformed_df = process_data(default_input_path, default_output_path)
        logging.info("--- Script finished successfully ---")
    except FileNotFoundError as e:
        logging.error(f"Execution failed: {e}")
        logging.info("Please ensure the data.csv file is in the correct directory.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}", exc_info=True)

