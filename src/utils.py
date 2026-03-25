import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, name="Model"):
    """
    Compute and print MAE, RMSE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {name} Results ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def log_experiment(results, config, path="experiment_log.txt"):
    """
    Log experimental results to a file.
    """
    with open(path, "a") as f:
        f.write(f"\nConfig: {config}\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write("-" * 20 + "\n")
