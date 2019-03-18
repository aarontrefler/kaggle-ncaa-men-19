"""Feature creation"""
import pandas as pd


def create_advanced_statistics(df):
    """
    Create advanced statistics by combing existing default statistics
    Code based on code from Kaggle public notebook: "ncaa-2k19-eda-zion-s-kingdom"
    """
    df['Poss']=df['FGA'] + 0.475*df['FTA'] - df['OR'] + df['TO']
    df['Poss_opp']=df['FGA_opp'] + 0.475*df['FTA_opp'] - df['OR_opp'] + df['TO_opp']
    df['OffRating']=100*(df['Score'] / df['Poss'])
    df['DefRating']=100*(df['Score_opp'] / df['Poss_opp'])
    df['NetRating']=df['OffRating'] - df['DefRating']
    df['Pace']=48*((df['Poss']+df['Poss_opp'])/(2*(240/5)))
    
    return df


def create_diff_feats(df):
    """Create features based on differences of existing features"""
    feats = [
        'Ast', 'Blk', 'DR', 'FGA', 'FGA3', 'FGM', 'FGM3', 'FTA', 'FTM', 'OR', 'PF', 'Score', 'Stl', 'TO',
        'Poss'
    ]
    feats_no_opp = [
        'Seed', 'NumOT',
        'OffRating', 'DefRating', 'NetRating', 'Pace'
    ]
    
    for feat in feats + feats_no_opp:
        df['Diff_' + feat] = eval('df.TeamOne_{feat} - df.TeamTwo_{feat}'.format(feat=feat))
        if feat not in feats_no_opp:
            df['Diff_' + feat + '_opp'] = eval('df.TeamOne_{feat}_opp - df.TeamTwo_{feat}_opp'.format(feat=feat))
                        
    return df