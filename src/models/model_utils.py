"""Machine learning models"""
import numpy as np

nonfeatures = ['Label', 'Season', 'TeamOneID', 'TeamTwoID', 'TeamOne_Name', 'TeamTwoName']

def clip_preds(y):
    """Restrict maximum and minimum value for probability prediction"""
    return np.clip(y, 0.05, 0.95)


def compute_logloss(df):
    """Create negative log loss column"""
    df['Logloss']= -1 * (df.Label * np.log(df.Pred) + (1 - df.Label) * np.log(1 - df.Pred))
    return df


def train_validation_split(df):
    """Temporally split training data into training and validation sets"""
    df_train = df.loc[df.Season.isin(np.arange(1985, 2014))]
    df_valid = df.loc[df.Season.isin(np.arange(2014, 2019))]

    return df_train, df_valid


def get_features(df):
    """Return list of feature columns"""
    cols_to_drop = [col for col in nonfeatures if col in df]

    return df.drop(cols_to_drop, axis=1).columns


def create_ID(row):
    """Create formatted identifier for submission file"""
    return "{season}_{teamOne}_{teamTwo}".format(
        season=int(row.Season), teamOne=int(row.TeamOneID), teamTwo=int(row.TeamTwoID))
