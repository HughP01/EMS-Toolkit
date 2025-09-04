import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_dataframe_summary(df, sample_size=5):
    """
    Generate a comprehensive summary of a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to summarize
    sample_size (int): Number of sample rows to display (default: 5)
    
    Returns:
    dict: A dictionary containing various summary statistics
    """
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    summary = {}
    
    # Basic information
    summary['shape'] = df.shape
    summary['dimensions'] = f"{df.shape[0]} rows × {df.shape[1]} columns"
    summary['memory_usage'] = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    
    # Column information
    summary['columns'] = {
        'names': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'non_null_counts': df.count().to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    }
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts().to_dict()
    summary['dtype_distribution'] = {str(k): v for k, v in dtype_counts.items()}
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'top_frequency': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
            }
        summary['categorical_stats'] = categorical_stats
    
    # Duplicate information
    summary['duplicates'] = {
        'total_duplicates': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df) * 100).round(2)
    }
    
    # Sample data
    summary['sample_data'] = df.head(sample_size).to_dict('records')
    
    return summary

def summary(df, sample_size=5):
    """
    Print a formatted summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to summarize
    sample_size (int): Number of sample rows to display
    """
    
    summary = generate_dataframe_summary(df, sample_size)

    print("DATAFRAME SUMMARY")
    
    print(f"\n BASIC INFORMATION:")
    print(f"   Shape: {summary['shape']}")
    print(f"   Dimensions: {summary['dimensions']}")
    print(f"   Memory Usage: {summary['memory_usage']}")
    
    print(f"\n DATA TYPES DISTRIBUTION:")
    for dtype, count in summary['dtype_distribution'].items():
        print(f"   {dtype}: {count} columns")
    
    print(f"\n NULL VALUES SUMMARY:")
    for col, null_pct in summary['columns']['null_percentages'].items():
        null_count = summary['columns']['null_counts'][col]
        if null_count == 0:
            continue
        else:
            print(f"   {col}: {null_count} null values ({null_pct}%)")
    
    print(f"\n NUMERIC COLUMNS STATISTICS:")
    if 'numeric_stats' in summary:
        for col in summary['numeric_stats'].keys():
            stats = summary['numeric_stats'][col]
            print(f"   {col}:")
            print(f"     Count: {stats['count']:.0f}, Mean: {stats['mean']:.2f},Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Std: {stats['std']:.2f}")
    else:
        print("   No numeric columns found")
    
    print(f"\n CATEGORICAL COLUMNS STATISTICS:")
    if 'categorical_stats' in summary:
        for col, stats in summary['categorical_stats'].items():
            print(f"   {col}:")
            print(f"     Unique values: {stats['unique_values']}")
            print(f"     Most frequent: '{stats['top_value']}' ({stats['top_frequency']} times)")
    else:
        print("   No categorical columns found")
    
    print(f"\n DUPLICATE ROWS:")
    print(f"   Total duplicates: {summary['duplicates']['total_duplicates']}")
    print(f"   Duplicate percentage: {summary['duplicates']['duplicate_percentage']}%")
    
    print(f"\n SAMPLE DATA (first {sample_size} rows):")
    sample_df = pd.DataFrame(summary['sample_data'])
    print(sample_df.to_string(index=False))

def correlation(dataframe,method='pearson'):
    """
    just gives numeric feedback
    in:
        pandas df
        correlation method (opt)
    returns:
        prints corr matrix
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    #only numeric columns
    numeric_df = dataframe.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame has no numeric columns to compute correlation.")
    
    corr = numeric_df.corr(method=method)
    print(corr)


#Missing and Dupe Data Check
def data_check(data):
    """
    Check for missing values and duplicates in a pandas DataFrame.
    
    Parameters:
        data (pd.DataFrame): The data to check.
    
    Returns:            
        Prints the results of the checks.
    """
    missing_total = data.isnull().sum().sum()
    duplicates_total = data.duplicated().sum()

    print(f"Missing values: {missing_total:,} {'Warning. Checking advised.' if missing_total > 0 else '✅'}")
    print(f"Duplicates: {duplicates_total:,} {'Warning. Checking advised.' if duplicates_total > 0 else '✅'}")

    if missing_total > 0:
        missing_cols = data.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        print(f"Columns with missing values: {', '.join(missing_cols.index)}")

    if duplicates_total > 0:
        dupe_cols = data.duplicated().sum()
        dupe_cols = dupe_cols[dupe_cols > 0]
        print(f"Columns with duplicate values: {', '.join(dupe_cols.index)}")

#Identify Potential Outliers
def find_outliers(df, show_rows=False, show_details=False):
    """
    Find outliers in each column of a pandas DataFrame using the IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to analyze for outliers
    show_rows : bool, optional
        If True, displays the actual outlier rows for each column
    show_details : bool, optional
        If True, shows detailed information including fences and quartiles
    """
    
    results = {}
    
    for column in df.select_dtypes(include=[np.number]).columns:  # Only numeric columns
        # Calculate quartiles and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate fences
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers_mask = (df[column] < lower_fence) | (df[column] > upper_fence)
        outlier_count = outliers_mask.sum()
        outlier_rows = df[outliers_mask]
        
        # Store results
        column_result = {
            'outlier_count': outlier_count,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'outlier_rows': outlier_rows
        }
        results[column] = column_result
    
    # Display results
    print("OUTLIER ANALYSIS REPORT")
    for column, result in results.items():
        if result['outlier_count'] == 0:
            continue #Only shows columns with outliers 
        else:
            print(f"\n Column: {column}")
            print(f"   Outliers found: {result['outlier_count']}")
        if show_details:
            print(f"   Q1: {result['Q1']:.4f}")
            print(f"   Q3: {result['Q3']:.4f}")
            print(f"   IQR: {result['IQR']:.4f}")
            print(f"   Lower fence: {result['lower_fence']:.4f}")
            print(f"   Upper fence: {result['upper_fence']:.4f}")
        
        if show_rows and result['outlier_count'] > 0:
            print(f"   Outlier values:")
            # Show just the outlier values and their index
            for idx, value in result['outlier_rows'][column].items():
                print(f"      Index {idx}: {value:.4f}")
        
        if show_rows and result['outlier_count'] > 5:  # Show preview if many outliers
            print(f"   ... ({result['outlier_count'] - 5} more outliers)")
