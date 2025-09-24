import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def dot_plot(data, x_col, y_col, hue_col=None, title="Scatter Plot", 
                 x_label=None, y_label=None, figsize=(10, 6), alpha=0.7, 
                 palette='viridis', save_path=None, reg_line=False,
                 ci=95, reg_color='red', reg_alpha=0.5,
                 reg_width=2):
    """
    Create a scatter plot using Seaborn with optional regression line.
    
    Parameters:
    -----------
    data : pandas DataFrame
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
        ax = sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, 
                           alpha=alpha, palette=palette)
    else:
        ax = sns.scatterplot(data=data, x=x_col, y=y_col, 
                           alpha=alpha, palette=palette)
    
    #regression line if requested
    if reg_line:
        sns.regplot(data=data, x=x_col, y=y_col, scatter=False, 
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

def world_chart(data, countries, column, 
                        locationmode='country names',
                        color_scale='Viridis',
                        title=None,
                        width=1000,
                        height=600):
    """
    Choropleth function with customization options.
    
    Parameters:
    locationmode: 'country names', 'ISO-3', or 'USA-states'
    color_scale: Any Plotly color scale ('Viridis', 'Plasma', 'Reds', etc.)
    """
    
    if title is None:
        title = f'World Map - {column} Distribution'
    
    fig = px.choropleth(
        data,
        locations=countries,
        locationmode=locationmode,
        color=column,
        hover_name=countries,
        color_continuous_scale=color_scale,
        title=title,
        labels={column: column.replace('_', ' ').title()},
        width=width,
        height=height
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def show_world_chart(data,country,column):
    """
    generates and displays the choropleth chart
    """
    fig = world_chart(data, country, column)
    fig.show()

def lineplot(data, x_col, y_col, hue_col=None, title="Line Chart", 
                   x_label=None, y_label=None, figsize=(10, 6), alpha=0.8, 
                   palette='viridis', save_path=None, ci=95, 
                   style_col=None, markers=False, linewidth=2.5,
                   err_style='band', err_alpha=0.3):
    """
    Create a line chart using Seaborn with optional confidence intervals.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The DataFrame containing the data to plot
    x_col : str
        Column name for the x-axis (should be numeric or datetime)
    y_col : str
        Column name for the y-axis
    hue_col : str, optional
        Column name for color encoding (categorical variable)
    title : str, optional
        Title of the plot (default: "Line Chart")
    x_label : str, optional
        Label for x-axis (default: uses x_col)
    y_label : str, optional
        Label for y-axis (default: uses y_col)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6))
    alpha : float, optional
        Transparency of lines (default: 0.8)
    palette : str, optional
        Color palette name (default: 'viridis')
    save_path : str, optional
        Path to save the plot (e.g., 'plot.png')
    ci : int, None, or 'sd', optional
        Confidence interval size:
        - int: Confidence interval size (e.g., 95 for 95% CI)
        - None: No confidence interval
        - 'sd': Show standard deviation instead of CI
    style_col : str, optional
        Column name for line style encoding
    markers : bool, optional
        Whether to show markers on data points (default: False)
    linewidth : float, optional
        Width of the lines (default: 2.5)
    err_style : str, optional
        Style of error representation: 'band' or 'bars' (default: 'band')
    err_alpha : float, optional
        Transparency of error region (default: 0.3)
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    #figure and axes
    plt.figure(figsize=figsize)
    
    #line plot
    ax = sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, style=style_col,
                     ci=ci, err_style=err_style, markers=markers, 
                     alpha=alpha, palette=palette, linewidth=linewidth)
    
    #labels and title
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(x_label if x_label else x_col)
    ax.set_ylabel(y_label if y_label else y_col)
    
    #error region transparency if CI is shown
    if ci is not None:
        for collection in ax.collections:
            collection.set_alpha(err_alpha)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if they're long (useful for datetime)
    if data[x_col].dtype == 'object' or hasattr(data[x_col].dtype, 'tz'):
        plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax

  
    
