#This file is for basic charts that can be used for basic data exploration and visualasation not to be confused with "graphing.py" which is intended for more advanced graphing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#corr chart 
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
