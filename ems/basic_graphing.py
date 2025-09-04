#This file is for basic charts that can be used for basic data exploration and visualasation not to be confused with "graphing.py" which is intended for more advanced graphing

import pandas as pd
import numpy as np
from scipy import stats
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


def analyze_dist(df, figsize=(15, 10)):
    """
    Analyze frequency counts and distributions for all columns in a pandas DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    figsize (tuple): Figure size for the plots (width, height)
    """
    
    #plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)
    
    #Get column information
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("ANALYSIS REPORT")
    print(f"Total rows: {len(df)}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    
    #numeric columns
    if numeric_cols:
        print("\nNUMERIC COLUMNS ANALYSIS:")
        
        #grid size for numeric plots
        n_numeric = len(numeric_cols)
        n_cols = min(3, n_numeric)
        n_rows = (n_numeric + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_numeric > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                #Histogram (with KDE)
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    #categorical columns
    if categorical_cols:
        print("\nCATEGORICAL COLUMNS ANALYSIS:")
        
        #Determine grid size for categorical plots
        n_categorical = len(categorical_cols)
        n_cols = min(2, n_categorical)
        n_rows = (n_categorical + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        axes = axes.flatten() if n_categorical > 1 else [axes]
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Count plot
                value_counts = df[col].value_counts()
                
                #For columns with many unique values, show top 10
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    sns.barplot(y=top_values.values, x=top_values.index, ax=axes[i])
                    axes[i].set_title(f'Top 10 values in {col}')
                else:
                    sns.barplot(y=value_counts.values, x=value_counts.index, ax=axes[i])
                    axes[i].set_title(f'Value counts for {col}')
                
                axes[i].set_xlabel('Count')
                axes[i].set_ylabel(col)
                
        
        #Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()
