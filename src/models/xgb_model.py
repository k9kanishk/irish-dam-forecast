from xgboost import XGBRegressor


def make_model():
    """
    XGBoost model optimized for faster training in dashboard context.
    
    Changes from original:
    - n_estimators: 1200 -> 300 (4x faster)
    - learning_rate: 0.03 -> 0.08 (compensates for fewer trees)
    - early_stopping ready
    """
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
    )
