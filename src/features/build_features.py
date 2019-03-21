"""Create dataset of features and meta-data to be used for modeling"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import features.feature_utils as feat
import utils

# Setup
data_interim_dir = utils.project_path + 'data/interim/'
data_clean_dir = utils.project_path + 'data/clean/'

# Read interim datasets
df_interim_mdl = (
    pd.read_csv(data_interim_dir + 'model_dataset.csv')
    .drop(['TeamOne_Name', 'TeamTwo_Name'], axis=1)
)
df_interim_sub = (
    pd.read_csv(data_interim_dir + 'submission_dataset.csv')
    .drop(['TeamOne_Name', 'TeamTwo_Name'], axis=1)
)

# Create features for modeling dataset
df_clean_mdl = (
    df_interim_mdl
    .pipe(feat.create_diff_feats)
    .pipe(feat.impute_missing_values, imputer=SimpleImputer(missing_values=np.nan, strategy='median'))
)
# Create features for submission dataset
df_clean_sub = (
    df_interim_sub
    .pipe(feat.create_diff_feats)
    .pipe(feat.impute_missing_values, imputer=SimpleImputer(missing_values=np.nan, strategy='median'))  # no missing data, performed for float conversion consistencty
)

# Create clean datasets
df_clean_mdl_save = (
    df_clean_mdl
    .sort_values(['Season', 'TeamOneID'])
    .pipe(utils.cols_to_front, front_cols=['Season'])
)
df_clean_sub_save = (
    df_clean_sub
    .sort_values(['Season', 'TeamOneID'])
    .pipe(utils.cols_to_front, front_cols=['Season'])
)

# Save clean datasets
df_clean_mdl_save.to_csv(data_clean_dir + 'model_dataset.csv', index=False)
df_clean_sub_save.to_csv(data_clean_dir + 'submission_dataset.csv', index=False)
