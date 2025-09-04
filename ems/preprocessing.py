import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats.mstats import winsorize

def handle_missing_values(df, strategy='avg', columns=None):
    #Implementation here
    """
    df: pandas DataFrame
    strategy: 'avg' replaces with mean for numeric columns median for categorical, 'drop' removes rows with missing values
    columns: list of columns to apply the strategy on, if None applies to all columns
    """
    print("Under Construction")

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep='first')

def drop_columns(df, columns_to_drop):
    """Drop specified columns from DataFrame."""
    return df.drop(columns=columns_to_drop, errors='ignore')

#Data Transforlamtion 
def standardize(df, columns=None):
    """
    Standardize numeric columns (z-score normalization).
    Returns DataFrame with standardized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def normalize(df, columns=None):
    """
    Min-max normalization (scale to 0-1 range).
    Returns DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def log_transform(df, columns, add_one=True):
    """
    Apply log transformation to specified columns.
    add_one: Add 1 to avoid log(0) if needed.
    """
    for col in columns:
        if add_one:
            df[col] = np.log1p(df[col])
        else:
            df[col] = np.log(df[col])
    return df

#Outlier Handling
def remove_outliers_iqr(df, columns=None, threshold=1):
    """
    Remove outliers using Interquartile Range method (Quartiles +- IQR).
    Returns DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def winsorize(df, columns=None, limits=[0.05, 0.05]):
    """
    Winsorize outliers by capping at specified percentiles.
    """
    #implementation here

    print("Under Construction")

def convert(df, columns, target_dtype, errors='ignore', inplace=True):
    """
    Convert specified columns to the desired data type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : str or list
        Single column name or list of column names to convert
    target_dtype : str
        Desired data type: 'string', 'float', 'int', 'object', 'category', 'datetime'
    errors : str, default 'ignore'
        How to handle errors: 'raise', 'coerce', 'ignore'
    inplace : bool, default True
        If True, modify the dataframe in place
    
    Returns:
    --------
    pandas.DataFrames
    """
    
    dtype_mapping = {
        'string': 'string',
        'str': 'string',
        'object': 'object',
        'float': 'float64',
        'int': 'int64',
        'integer': 'int64',
        'category': 'category',
        'datetime': 'datetime64[ns]'
    }
    
    if target_dtype.lower() not in dtype_mapping:
        raise ValueError(f"Unsupported target_dtype: {target_dtype}")
    
    if isinstance(columns, str):
        columns = [columns]
    
    if not inplace:
        df = df.copy()
    
    target_pandas_dtype = dtype_mapping[target_dtype.lower()]
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
            
        try:
            if target_dtype.lower() in ['float', 'int', 'integer']:
                df[col] = pd.to_numeric(df[col], errors=errors)
            elif target_dtype.lower() == 'datetime':
                df[col] = pd.to_datetime(df[col], errors=errors)
            
            df[col] = df[col].astype(target_pandas_dtype, errors=errors)
            print(f"Converted '{col}' to {target_pandas_dtype}")
            
        except Exception as e:
            print(f"Error converting '{col}': {e}")
            if errors == 'raise':
                raise
    
    return df
