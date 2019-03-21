"""Make predcitions and create submission file"""
import pickle

import pandas as pd

import models.model_utils as model
import utils

# Setup
data_clean_dir = utils.project_path + 'data/clean/'

# Read data
df_clean_sub = pd.read_csv(data_clean_dir + 'submission_dataset.csv')

# Read model
f = open(utils.project_path + 'models/gridcv.model', 'rb')
gridcv = pickle.load(f)
f.close()

# Define features
features = model.get_features(df_clean_sub)

# Make submission set predictions
X_sub = df_clean_sub[features].values
df_clean_sub_pred = (
    df_clean_sub
    .assign(Pred=model.clip_preds(gridcv.best_estimator_.predict_proba(X_sub)[:, 1]))
    .pipe(utils.cols_to_front, front_cols=['Pred'])
)

# Save submission set predictions
df_clean_sub_pred.to_csv(data_clean_dir + 'gridcv_submission_dataset_predictions.csv', index=False)

 # Create submission file
df_submsission = (
    df_clean_sub_pred
    .assign(ID=df_clean_sub.apply(model.create_ID, axis=1))
    .loc[:, ['ID', 'Pred']]
)

# Save submssion file
datestamp = utils.create_datestamp()
df_submsission.to_csv(utils.project_path + 'models/submissions/{name}_submission.csv'.format(
    name='gridcv_' + datestamp), index=False)
