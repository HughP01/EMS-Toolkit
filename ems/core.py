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