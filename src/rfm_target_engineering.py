import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rfm_metrics(df, snapshot_date):
    """Calculate RFM metrics for each CustomerId."""
    logging.info("Calculating RFM metrics...")
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    # Recency: Days since last transaction
    recency_df = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency_df['Recency'] = (snapshot_date - recency_df['TransactionStartTime']).dt.days
    
    # Frequency: Number of transactions
    frequency_df = df.groupby('CustomerId').size().reset_index(name='Frequency')
    
    # Monetary: Sum of absolute Amount
    monetary_df = df.groupby('CustomerId')['Amount'].apply(lambda x: np.abs(x).sum()).reset_index(name='Monetary')
    
    # Combine RFM
    rfm_df = recency_df[['CustomerId', 'Recency']].merge(
        frequency_df, on='CustomerId').merge(
        monetary_df, on='CustomerId')
    
    logging.info(f"RFM metrics summary:\n{rfm_df[['Recency', 'Frequency', 'Monetary']].describe().to_string()}")
    return rfm_df

def cluster_rfm(rfm_df):
    """Cluster customers into 3 groups and assign is_high_risk."""
    logging.info("Clustering RFM data...")
    
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Analyze clusters
    cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    logging.info(f"Cluster summary:\n{cluster_summary.to_string()}")
    
    # Identify high-risk cluster (lowest Frequency)
    high_risk_cluster = cluster_summary['Frequency'].idxmin()
    logging.info(f"High-risk cluster: {high_risk_cluster}")
    
    # Assign is_high_risk
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    logging.info(f"is_high_risk distribution:\n{rfm_df['is_high_risk'].value_counts().to_string()}")
    
    return rfm_df[['CustomerId', 'is_high_risk']]

def process_rfm_target(input_path, processed_path, output_path):
    """Calculate RFM, cluster, and merge is_high_risk into processed dataset."""
    # Load raw data
    logging.info(f"Loading raw data from {input_path}...")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Raw data file {input_path} does not exist.")
    df = pd.read_csv(input_path)
    
    # Set snapshot date
    snapshot_date = pd.to_datetime(df['TransactionStartTime']).max()
    logging.info(f"Snapshot date: {snapshot_date}")
    
    # Calculate RFM and cluster
    rfm_df = calculate_rfm_metrics(df, snapshot_date)
    rfm_result = cluster_rfm(rfm_df)
    
    # Load processed dataset
    logging.info(f"Loading processed data from {processed_path}...")
    if not os.path.isfile(processed_path):
        raise FileNotFoundError(f"Processed data file {processed_path} does not exist.")
    processed_df = pd.read_csv(processed_path)
    
    # Check for CustomerId
    if 'remainder__CustomerId' not in processed_df.columns:
        logging.error(f"remainder__CustomerId not found in processed data. Available columns: {processed_df.columns.tolist()}")
        raise KeyError("remainder__CustomerId not found in processed dataset.")
    
    # Merge is_high_risk
    logging.info("Merging is_high_risk...")
    processed_df = processed_df.merge(rfm_result, left_on='remainder__CustomerId', right_on='CustomerId', how='left')
    processed_df = processed_df.drop(columns=['CustomerId'])  # Drop extra CustomerId from rfm_result
    if processed_df['is_high_risk'].isna().any():
        logging.warning("Missing is_high_risk values. Filling with 0.")
        processed_df['is_high_risk'] = processed_df['is_high_risk'].fillna(0).astype(int)
    
    # Save updated dataset
    logging.info(f"Saving updated dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    logging.info(f"Updated dataset saved to {output_path}")
    
    return processed_df

if __name__ == "__main__":
    default_input_path = '../data/raw/data.csv'
    default_processed_path = '../data/processed/transformed_transactions.csv'
    default_output_path = '../data/processed/transformed_transactions_with_risk.csv'
    try:
        transformed_df = process_rfm_target(
            default_input_path,
            default_processed_path,
            default_output_path
        )
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error: {str(e)}")