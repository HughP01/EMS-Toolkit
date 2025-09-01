"""
EMS - Extra Modelling System
A simple Python module to quickly perform data analysis.
Hugh Plunkett 2025 
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def correlation_chart(dataframe, method='pearson', figsize=(10,8), cmap='coolwarm'):
    """
    Generate a correlation heatmap from a pandas DataFrame.

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
    
    corr = dataframe.corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar=True, square=True)
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.tight_layout()
    plt.show()
