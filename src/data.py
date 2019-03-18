"""Data processing"""
import re

import numpy as np
import pandas as pd


def process_teams(df):
    """Process teams data file"""
    return  df.loc[:, ['TeamID', 'TeamName']]


def process_seeds(df):
    """Process tournament seed data file"""
    return (
        df
        .assign(Seed=df.Seed.map(lambda s: int(s[1:3])))
        .assign(IsPlayIn=df.Seed.map(lambda s: str(s).endswith('a') or str(s).endswith('b')))
    )


def process_regular_season_detailed_results(df):
    """Process regular season detailed results data file"""
    def _rename_winner_vs_opp(old_name):
        if re.match(r'^W', old_name):
            return re.sub('^W','', old_name)
        if re.match(r'^L', old_name):
            return re.sub('^L', '', old_name) + '_opp'
        return old_name
    
    def _rename_loser_vs_opp(old_name):
        if re.match(r'^L', old_name):
            return re.sub('^L','', old_name)
        if re.match(r'^W', old_name):
            return re.sub('^W', '', old_name) + '_opp'
        return old_name
    
    df_winners = df.rename(columns=_rename_winner_vs_opp)
    df_losers = df.rename(columns=_rename_loser_vs_opp)
    
    return pd.concat([df_winners, df_losers])


def process_games(df):
    """Process tournament game data file"""
    
    def _standardize_team_ID(row):
        """Inner function to sandardize team ID"""
        if row.WTeamID < row.LTeamID:
            row.TeamOneID = row.WTeamID
            row.TeamTwoID = row.LTeamID
        else:
            row.TeamOneID = row.LTeamID
            row.TeamTwoID = row.WTeamID
        return row

    def _create_labels(row):
        """Inner function to assign correct label"""
        if row.TeamOneID == row.WTeamID:
            row.Label = 1
        else:
            row.Label = 0
        return row
    
    return (
        df
        .assign(
            TeamOneID=np.nan,
            TeamTwoID=np.nan,
            Label=np.nan
        )
        .apply(_standardize_team_ID, axis=1)
        .apply(_create_labels, axis=1)
        .drop(['WTeamID', 'LTeamID', 'DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
    )


def process_submission_games(df):
    """Process tournament game submission file"""
    return (
        df
        .assign(
            Season=df.ID.str.split('_', expand=True)[0].astype(int),    
            TeamOneID=df.ID.str.split('_', expand=True)[1].astype(int),
            TeamTwoID=df.ID.str.split('_', expand=True)[2].astype(int),
    )
    .drop(['ID', 'Pred'], axis=1)
)


def merge_seed_dataset(df, df_seeds, submission_file):
    """Merge tournament games with seed dataset"""
    df_out = (
        df
        .merge(df_seeds.rename(index=str, columns={'TeamID': 'TeamOneID'}), on=['Season', 'TeamOneID'])
        .rename(index=str, columns={'Seed': 'TeamOne_Seed', 'IsPlayIn': 'TeamOneIsPlayIn'})
        .merge(df_seeds.rename(index=str, columns={'TeamID': 'TeamTwoID'}), on=['Season', 'TeamTwoID'])
        .rename(index=str, columns={'Seed': 'TeamTwo_Seed', 'IsPlayIn': 'TeamTwoIsPlayIn'})  
    )
    
    if not(submission_file):
        df_out = df_out.loc[~(df_out.TeamOneIsPlayIn & df_out.TeamTwoIsPlayIn)]  # remove play-in games 
    
    return df_out.drop(['TeamOneIsPlayIn', 'TeamTwoIsPlayIn'], axis=1)


def merge_aggregated_regular_season_detailed_results(df, df_rs_d_res_agg):
    """Merge tournament games with regular season detaield results dataset"""
    def _suffix_to_prefix(old_name):
        if old_name.endswith('_TeamOne'):
            return 'TeamOne_{}'.format(re.sub('_TeamOne', '', old_name))
        if old_name.endswith('_TeamTwo'):
            return 'TeamTwo_{}'.format(re.sub('_TeamTwo', '', old_name))
        return old_name
    
    return (
        df
        .merge(df_rs_d_res_agg.rename(index=str, columns={'TeamID': 'TeamOneID'}), on=['Season', 'TeamOneID'], 
               how='left')  # team one data 
        .merge(df_rs_d_res_agg.rename(index=str, columns={'TeamID': 'TeamTwoID'}), on=['Season', 'TeamTwoID'], 
               how='left', suffixes=('_TeamOne', '_TeamTwo'))  # team two data
        .rename(columns=_suffix_to_prefix)
    )


def merge_team_dataset(df, df_teams):
    """Merge tournament games with team dataset"""    
    return (
        df
        .merge(df_teams.rename(index=str, columns={'TeamID': 'TeamOneID', 'TeamName': 'TeamOne_Name'}), on='TeamOneID')
        .merge(df_teams.rename(index=str, columns={'TeamID': 'TeamTwoID', 'TeamName': 'TeamTwo_Name'}), on='TeamTwoID')
    )

