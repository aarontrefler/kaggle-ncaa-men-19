"""Machine learning models"""
import numpy as np

def clip_preds(y):
    """Restrict maximum and minimum value for probability prediction"""
    return np.clip(y, 0.05, 0.95)


def compute_logloss(df):
    """Create negative log loss column"""
    df['Logloss']= -1 * (df.Label * np.log(df.Pred) + (1 - df.Label) * np.log(1 - df.Pred))
    return df