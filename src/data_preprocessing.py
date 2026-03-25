import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """
    Load data from CSV and apply filtering rules from Section IV-B.
    """
    df = pd.read_csv(file_path)
    
    # Convert datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    # Compute trip duration in minutes
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filtering rules
    df = df[df['trip_distance'] > 0]
    df = df[df['fare_amount'] > 0]
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['trip_duration'] >= 1) & (df['trip_duration'] <= 120)]
    df = df[df['total_amount'] >= 0]
    
    return df.reset_index(drop=True)

def feature_engineering(df):
    """
    Derive temporal, economic, and kinematic features as per Section V.
    """
    # Temporal features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclic encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    
    # Economic features
    df['log_trip_distance'] = np.log1p(df['trip_distance'])
    df['fare_per_mile'] = df['fare_amount'] / (df['trip_distance'] + 1e-6)
    df['tip_rate'] = df['tip_amount'] / (df['fare_amount'] + 1e-6)
    
    # Kinematic features (Equation 2)
    # avg_speed = distance / (duration_min/60 + 1e-6)
    df['avg_speed'] = df['trip_distance'] / (df['trip_duration'] / 60 + 1e-6)
    
    # Handle NaNs/Infs that might have occurred during division
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def prepare_tabular_data(df, selected_features):
    """
    Prepare X and y for baseline models.
    """
    X = df[selected_features]
    y = df['trip_duration']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, scaler

def prepare_gnn_data(df, src_col='PULocationID', dst_col='DOLocationID'):
    """
    Prepare mapping and features for GraphSAGE.
    """
    node_list = sorted(pd.concat([df[src_col], df[dst_col]]).unique())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)
    
    # Aggregate node features (mean dist, fare, duration, log-count)
    node_stats = df.groupby(src_col).agg({
        'trip_distance': 'mean',
        'fare_amount': 'mean',
        'trip_duration': 'mean'
    }).reindex(node_list).fillna(0)
    
    # Log-count of trips per zone
    counts = pd.concat([df[src_col], df[dst_col]]).value_counts().reindex(node_list).fillna(1)
    node_stats['log_count'] = np.log1p(counts)
    
    node_features = StandardScaler().fit_transform(node_stats.values).astype(np.float32)
    
    # Edge preparation
    edge_src = df[src_col].map(node_to_idx).values
    edge_dst = df[dst_col].map(node_to_idx).values
    
    return node_features, edge_src, edge_dst, node_to_idx
