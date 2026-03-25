import os
import torch
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    load_and_clean_data, 
    feature_engineering, 
    prepare_tabular_data, 
    prepare_gnn_data
)
from src.training import train_baseline, train_gnn
from src.models import GraphSAGENet, EdgeMLP
from src.utils import log_experiment

# ---------------- Configuration ----------------
CONFIG = {
    'file_path': 'ML_AAT.csv',  # Update this to your actual data path
    'random_seed': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'epochs': 80,
    'batch_size': 2048,
    'gnn_hidden': 128,
    'lr': 1e-4,
    'clip_norm': 1.0,
    'test_size': 0.2
}

# Set seeds
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def main():
    if not os.path.exists(CONFIG['file_path']):
        print(f"Error: Data file {CONFIG['file_path']} not found.")
        print("Please ensure your dataset is in the current directory.")
        return

    # 1. Data Pipeline
    print("--- Starting Data Pipeline ---")
    df = load_and_clean_data(CONFIG['file_path'])
    df = feature_engineering(df)
    print(f"Data shape after cleaning: {df.shape}")

    # 2. Baseline Models
    print("\n--- Training Baseline Models ---")
    selected_features = [
        'trip_distance', 'fare_amount', 'passenger_count', 'pickup_hour',
        'hour_sin', 'hour_cos', 'tip_amount', 'tolls_amount', 'total_amount',
        'day_of_week', 'is_weekend', 'log_trip_distance', 'fare_per_mile',
        'tip_rate', 'avg_speed'
    ]
    X_scaled, y, scaler = prepare_tabular_data(df, selected_features)
    
    # Split
    split_idx = int(len(X_scaled) * (1 - CONFIG['test_size']))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    baselines = ['lr', 'knn', 'dt', 'rf', 'xgboost']
    baseline_results = {}
    for model_type in baselines:
        _, metrics = train_baseline(X_train, y_train, X_test, y_test, model_type=model_type)
        baseline_results[model_type] = metrics

    # 3. GNN Prediction (Experimental)
    print("\n--- Training GraphSAGE (GNN) ---")
    node_feat, edge_src, edge_dst, node_to_idx = prepare_gnn_data(df)
    
    # Prepare PyTorch tensors
    node_x = torch.tensor(node_feat, dtype=torch.float32).to(CONFIG['device'])
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(CONFIG['device'])
    # Undirected edges for message passing
    edge_index_ud = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Edge attributes for MLP
    edge_attr_cols = ['trip_distance', 'fare_amount', 'hour_sin', 'hour_cos', 'tip_rate', 'avg_speed']
    edge_attr = torch.tensor(df[edge_attr_cols].values, dtype=torch.float32).to(CONFIG['device'])
    
    # Target: log-duration for stability
    y_edge = torch.tensor(np.log1p(df['trip_duration'].values), dtype=torch.float32).view(-1, 1).to(CONFIG['device'])
    
    # GNN Model Initialization
    gnn = GraphSAGENet(node_x.shape[1], CONFIG['gnn_hidden']).to(CONFIG['device'])
    edge_mlp = EdgeMLP(CONFIG['gnn_hidden'], edge_attr.shape[1]).to(CONFIG['device'])
    
    # Indices for edge training
    perm = np.random.permutation(len(df))
    train_slice = int(len(df) * (1 - CONFIG['test_size']))
    train_idx = perm[:train_slice]
    test_idx = perm[train_slice:]
    
    gnn_metrics = train_gnn(
        gnn, edge_mlp, node_x, edge_index_ud, edge_index, edge_attr, y_edge, 
        train_idx, test_idx, CONFIG
    )
    
    # 4. Cleanup & Logging
    print("\nProject Refactor Complete!")
    log_experiment({"baselines": baseline_results, "gnn": gnn_metrics}, CONFIG)

if __name__ == "__main__":
    main()