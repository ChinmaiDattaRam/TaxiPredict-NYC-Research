import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .utils import calculate_metrics

def train_baseline(X_train, y_train, X_test, y_test, model_type="xgboost", params=None):
    """
    Train and evaluate a baseline regressor.
    """
    if model_type == "lr":
        model = LinearRegression()
    elif model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == "dt":
        model = DecisionTreeRegressor(max_depth=14, random_state=42)
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    elif model_type == "xgboost":
        default_params = {
            "n_estimators": 400,
            "learning_rate": 0.07,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42
        }
        if params:
            default_params.update(params)
        model = XGBRegressor(**default_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, name=model_type.upper())
    return model, metrics

def train_gnn(gnn, edge_mlp, node_x, edge_index_ud, edge_index, edge_attr, y_edge, train_idx, test_idx, config):
    """
    Experimental GNN training loop (GraphSAGE).
    """
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 80)
    batch_size = config.get('batch_size', 2048)
    lr = config.get('lr', 1e-4)
    clip_norm = config.get('clip_norm', 1.0)

    params = list(gnn.parameters()) + list(edge_mlp.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    loss_fn = nn.HuberLoss()

    idx_train = torch.tensor(train_idx)
    
    for ep in range(epochs):
        gnn.train()
        edge_mlp.train()
        perm = torch.randperm(len(idx_train))
        total_loss = 0
        
        for i in range(0, len(perm), batch_size):
            b_idx = idx_train[perm[i : i + batch_size]]
            
            # Forward pass
            emb = gnn(node_x, edge_index_ud)
            src_emb = emb[edge_index[0, b_idx]]
            dst_emb = emb[edge_index[1, b_idx]]
            ea = edge_attr[b_idx]
            
            pred = edge_mlp(src_emb, dst_emb, ea)
            target = y_edge[b_idx]
            
            loss = loss_fn(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
            optimizer.step()
            
            total_loss += loss.item() * b_idx.size(0)
            
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1:3d} | Loss {total_loss/len(idx_train):.4f}")

    # Evaluation
    gnn.eval()
    edge_mlp.eval()
    with torch.no_grad():
        emb = gnn(node_x, edge_index_ud)
        src_emb = emb[edge_index[0, test_idx]]
        dst_emb = emb[edge_index[1, test_idx]]
        ea = edge_attr[test_idx]
        
        # Output is log-duration, expm1 to get real minutes
        pred_log = edge_mlp(src_emb, dst_emb, ea).cpu().numpy().ravel()
        y_pred = np.expm1(pred_log)
        y_test = np.expm1(y_edge[test_idx].cpu().numpy().ravel())
        
    metrics = calculate_metrics(y_test, y_pred, name="GNN (GraphSAGE)")
    return metrics
