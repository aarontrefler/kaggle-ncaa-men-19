"""Create trained classifier objects, modeling set predictions, and model report"""
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import models.model_utils as model
import utils

# Setup
data_clean_dir = utils.project_path + 'data/clean/'

# Read data
df_clean = pd.read_csv(data_clean_dir + 'model_dataset.csv')

# Cross validation split
df_train, df_valid = model.train_validation_split(df_clean)

# Define features
features = model.get_features(df_clean)

# Define grid search cv parameters
hyer_parameter_space = {
    'min_samples_split': [50, 75, 87, 100],
    'max_depth': [None],
    'n_estimators': [1000],
    'random_state': [3]
}
X = df_clean[features].values
y = df_clean.Label.values
clf = RandomForestClassifier()
ps_cv = PredefinedSplit(
    np.concatenate((np.zeros(len(df_train)) - 1, np.zeros(len(df_valid))))) # create custom cv object
scoring= ['neg_log_loss', 'accuracy']
refit= 'neg_log_loss'
# Perform grid search cv
gridcv = GridSearchCV(clf, hyer_parameter_space, 
    cv=ps_cv, scoring=scoring, refit=refit, error_score='raise', n_jobs=2, return_train_score=True
)
gridcv.fit(X, y)

# Fit model to training dataset with optimal hyper-parameters
X_train = df_train[features].values
y_train = df_train.Label.values
clf = RandomForestClassifier(**gridcv.best_params_).fit(X_train, y_train)
# Make training set predictions
df_train_pred = (
    df_train
    .assign(Pred=model.clip_preds(clf.predict_proba(X_train)[:, 1]))
    .pipe(utils.cols_to_front, front_cols=['Pred', 'Label'])
)
# Make validation set predictions
X_valid = df_valid[features].values
df_valid_pred = (
    df_valid
    .assign(Pred=model.clip_preds(clf.predict_proba(X_valid)[:, 1]))
    .pipe(utils.cols_to_front, front_cols=['Pred', 'Label'])
)
# Compute scores
logloss_train = log_loss(df_train_pred.Label, df_train_pred.Pred, normalize=True)
acc_train = sum(np.round(df_train_pred.Pred) == df_train_pred.Label) / len(df_train_pred)
logloss_valid = log_loss(df_valid_pred.Label, df_valid_pred.Pred, normalize=True)
acc_valid = sum(np.round(df_valid_pred.Pred) == df_valid_pred.Label) / len(df_valid_pred)

# Save model report
f = open('{project_path}models/reports/clf_{name}_report.txt'.format(
    project_path=utils.project_path, name=utils.create_datestamp()), 'w')
f.write('Random Forest Model\n')
f.write('Logloss Training:    {score:.5f}\n'.format(score=logloss_train))
f.write('Logloss Validation:  {score:.5f}\n'.format(score=logloss_valid))
f.write('Accuracy Training:   {score:.5f}\n'.format(score=acc_train))
f.write('Accuracy Validation: {score:.5f}\n\n'.format(score=acc_valid))
f.close()

# Save predictions
df_train_pred.to_csv(data_clean_dir + 'clf_training_dataset_predictions.csv', index=False)
df_valid_pred.to_csv(data_clean_dir + 'clf_validation_dataset_predictions.csv', index=False)

# Save grid search cv object that contains model fitted to entire dataset
f = open(utils.project_path + 'models/gridcv.model', 'wb')
pickle.dump(gridcv, f)
f.close()

# Save model fitted only to training data
f = open(utils.project_path + 'models/clf.model', 'wb')
pickle.dump(gridcv.best_estimator_, f)
f.close()
