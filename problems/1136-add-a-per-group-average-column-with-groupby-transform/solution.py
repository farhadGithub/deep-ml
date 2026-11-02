import pandas as pd

def solution(df):
    df['group_avg'] = df.groupby('group')['value'].transform('mean')
    return df