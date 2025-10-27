from xgboost import XGBRegressor


def make_model():
    return XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method='hist',
        random_state=42
    )
