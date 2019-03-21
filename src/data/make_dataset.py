"""Create a dataset from raw data files"""

import numpy as np
import pandas as pd

import data.data_utils as data
import utils

# Setup
data_raw_dir = utils.project_path + 'data/raw/'
data_interim_dir = utils.project_path + 'data/interim/'

# Read and process teams data
df_teams = (
    pd.read_csv(data_raw_dir + 'Stage2DataFiles/Teams.csv')
    .pipe(data.process_teams)
)
# Read and process seeds data
df_seeds = (
    pd.read_csv(data_raw_dir + 'Stage2DataFiles/NCAATourneySeeds.csv')
    .pipe(data.process_seeds)
)
# Read and process regular season detailed results data
df_rs_d_res = (
    pd.read_csv(data_raw_dir + 'Stage2DataFiles/RegularSeasonDetailedResults.csv')
    .pipe(data.process_regular_season_detailed_results)
    .pipe(data.create_advanced_statistics)
)
# Read and process tournament games data
df_games = (
    pd.read_csv(data_raw_dir + 'Stage2DataFiles/NCAATourneyCompactResults.csv')
    .pipe(data.process_games)
)
# Read and process tournament games submission file
df_sub = (
    pd.read_csv(data_raw_dir + 'SampleSubmissionStage2.csv')
    .pipe(data.process_submission_games)
)

# Aggregate regular season results dataset
df_rs_d_res_agg = (
    df_rs_d_res
    .groupby(['TeamID', 'Season'])
    .mean()
    .drop(['TeamID_opp', 'DayNum'], axis=1)
    .reset_index()  # re-assign Season from index to column
)

# Merge tournament games dataset with other datasets
df_interim_mdl = (
    df_games
    .pipe(data.merge_seed_dataset, df_seeds=df_seeds, submission_file=False)
    .pipe(data.merge_aggregated_regular_season_detailed_results, df_rs_d_res_agg=df_rs_d_res_agg)
    .pipe(data.merge_team_dataset, df_teams=df_teams)
)
# Merge submission file games dataset with other datasets
df_interim_sub = (
    df_sub
    .pipe(data.merge_seed_dataset, df_seeds=df_seeds, submission_file=True)
    .pipe(data.merge_aggregated_regular_season_detailed_results, df_rs_d_res_agg=df_rs_d_res_agg)
    .pipe(data.merge_team_dataset, df_teams=df_teams)
)

# Save datasets
front_cols = ['Season', 'TeamOne_Name', 'TeamTwo_Name', 'TeamOneID', 'TeamTwoID']
(
    df_interim_mdl
    .sort_values(['Season', 'TeamOneID', 'TeamTwoID'])
    .reset_index(drop=True)
    .pipe(utils.cols_to_front, front_cols=front_cols + ['Label'])
    .to_csv(data_interim_dir + 'model_dataset.csv', index=False)
)
(
    df_interim_sub
    .sort_values(['Season', 'TeamOneID', 'TeamTwoID'])
    .reset_index(drop=True)
    .pipe(utils.cols_to_front, front_cols=front_cols)
    .to_csv(data_interim_dir + 'submission_dataset.csv', index=False)
)
