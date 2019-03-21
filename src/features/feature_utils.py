"""Feature creation"""
import pandas as pd

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


def impute_missing_values(df, imputer):
    """Replace missing values in DataFrame"""
    return pd.DataFrame(imputer.fit_transform(df.values), columns=df.columns)
