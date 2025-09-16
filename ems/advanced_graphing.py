import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def dot_plot(df, x_col, y_col, hue_col=None, title="Scatter Plot", 
                 x_label=None, y_label=None, figsize=(10, 6), alpha=0.7, 
                 palette='viridis', save_path=None, reg_line=False,
                 ci=95, reg_color='red', reg_alpha=0.5,
                 reg_width=2):
    """
    Create a scatter plot using Seaborn with optional regression line.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the data to plot
    x_col : str
        Column name for the x-axis
    y_col : str
        Column name for the y-axis
    hue_col : str, optional
        Column name for color encoding (categorical variable)
    title : str, optional
        Title of the plot (default: "Scatter Plot")
    x_label : str, optional
        Label for x-axis (default: uses x_col)
    y_label : str, optional
        Label for y-axis (default: uses y_col)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6))
    alpha : float, optional
        Transparency of points (default: 0.7)
    palette : str, optional
        Color palette name (default: 'viridis')
    save_path : str, optional
        Path to save the plot (e.g., 'plot.png')
    reg_line : bool, optional
        Whether to add a regression line (default: False)
    ci : int or None, optional
        Confidence interval size (e.g., 95 for 95% CI). 
        Set to None to remove confidence interval.
    reg_color : str, optional
        Color of the regression line (default: 'red')
    reg_alpha : float, optional
        Transparency of confidence interval (default: 0.5)
    reg_width : float, optional
        Width of the regression line (default: 2)
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    # Create figure and axes
    plt.figure(figsize=figsize)
    
    #Create scatter plot
    if hue_col:
        ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                           alpha=alpha, palette=palette)
    else:
        ax = sns.scatterplot(data=df, x=x_col, y=y_col, 
                           alpha=alpha, palette=palette)
    
    #regression line if requested
    if reg_line:
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, 
                   ax=ax, color=reg_color, ci=ci,
                   line_kws={
                       'alpha': reg_alpha,
                       'linewidth': reg_width
                   })
    
    #labels and title
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(x_label if x_label else x_col)
    ax.set_ylabel(y_label if y_label else y_col)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    #save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax
