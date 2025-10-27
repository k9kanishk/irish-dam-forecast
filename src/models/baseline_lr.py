from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def make_model(alpha: float = 5.0):
    return Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('ridge', Ridge(alpha=alpha, random_state=42))
    ])
