import pandas as pd

def solution(df):
    cols = []
    for col in df.columns:
        if df[col].isna().mean()<=0.5:
            cols.append(col)
    df = df[cols]

    df = df[df.isna().mean(axis=1)<=0.5]

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df
    
    
   