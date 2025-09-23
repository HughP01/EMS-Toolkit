import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats.mstats import winsorize

def handle_missing_values(data, strategy='avg', columns=None):
    """
    Handle missing values in a pandas DataFrame.
    
    Parameters:
    data: pandas DataFrame
    strategy: 'avg' replaces with mean for numeric columns median for categorical, 'drop' removes rows with missing values
    columns: list of columns to apply the strategy on, if None applies to all columns
    
    Returns:
    pandas DataFrame with missing values handled
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # If columns is None, apply to all columns
    if columns is None:
        columns = df.columns.tolist()
    
    # Validate that specified columns exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if strategy == 'drop':
        # Drop rows with missing values in the specified columns
        df = df.dropna(subset=columns)
    
    elif strategy == 'avg':
        for col in columns:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric column - replace with mean
                    fill_value = df[col].mean()
                else:
                    # Categorical column - replace with median (mode for categorical)
                    # For categorical data, median doesn't make sense, so we use mode
                    if df[col].notna().sum() > 0:  # Check if there are non-null values
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    else:
                        fill_value = 'Unknown'
                
                df[col] = df[col].fillna(fill_value)
    
    else:
        raise ValueError("Strategy must be either 'avg' or 'drop'")
    
    return df

def remove_duplicates(data, subset=None):
    """Remove duplicate rows from DataFrame."""
    return data.drop_duplicates(subset=subset, keep='first')

def drop_columns(data, columns_to_drop):
    """Drop specified columns from DataFrame."""
    return data.drop(columns=columns_to_drop, errors='ignore')

#Data Transforlamtion 
def standardize(data, columns=None):
    """
    Standardize numeric columns (z-score normalization).
    Returns DataFrame with standardized columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data

def normalize(data, columns=None):
    """
    Min-max normalization (scale to 0-1 range).
    Returns DataFrame with normalized columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in data.columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

def log_transform(data, columns, add_one=True):
    """
    Apply log transformation to specified columns.
    add_one: Add 1 to avoid log(0) if needed.
    """
    for col in columns:
        if add_one:
            data[col] = np.log1p(data[col])
        else:
            data[col] = np.log(data[col])
    return data

#Outlier Handling
def remove_outliers_iqr(data, columns=None, threshold=1):
    """
    Remove outliers using Interquartile Range method (Quartiles +- IQR).
    Returns DataFrame with outliers removed.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    return data

def winsorize(data, columns=None, limits=[0.05, 0.05]):
    """
    Winsorize outliers by capping at specified percentiles.
    """
    #implementation here

    print("Under Construction")

def convert(data, columns, target_dtype, errors='ignore', inplace=True):
    """
    Convert specified columns to the desired data type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
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
        data = data.copy()
    
    target_pandas_dtype = dtype_mapping[target_dtype.lower()]
    
    for col in columns:
        if col not in data.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
            
        try:
            if target_dtype.lower() in ['float', 'int', 'integer']:
                data[col] = pd.to_numeric(data[col], errors=errors)
            elif target_dtype.lower() == 'datetime':
                data[col] = pd.to_datetime(data[col], errors=errors)
            
            data[col] = data[col].astype(target_pandas_dtype, errors=errors)
            print(f"Converted '{col}' to {target_pandas_dtype}")
            
        except Exception as e:
            print(f"Error converting '{col}': {e}")
            if errors == 'raise':
                raise
    
    return data

