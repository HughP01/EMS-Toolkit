import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#corr chart move to basic graphing eventually
def correlation_chart(dataframe, method='pearson', size=(10,8), cmap='coolwarm'):
    """
    Generate a correlation heatmap from a pandas DataFrame (numeric columns only).

    Parameters:
        dataframe (pd.DataFrame): The data to analyze.
        method (str): Correlation method ('pearson', 'kendall', 'spearman').
        figsize (tuple): Figure size for the plot.
        cmap (str): Seaborn colormap.

    Returns:

        matplotlib.figure.Figure: The heatmap figure.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    #only numeric columns
    numeric_df = dataframe.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("DataFrame has no numeric columns to compute correlation.")
    
    corr = numeric_df.corr(method=method)
    
    plt.figure(figsize=size)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar=True, square=True)
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.tight_layout()
    plt.show()

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
