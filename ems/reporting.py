from google import genai
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy import stats

def corr_report(dataframe, method='pearson'): #Correlation report generation
    """
    Generate a correlation report with AI-powered business insights
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        Input dataframe with numeric columns
    method : str, optional
        Correlation method ('pearson', 'kendall', 'spearman'), default 'pearson'
    
    Returns:
    --------
    str: AI-generated correlation analysis report
    """
    
    # Input validation
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    # Select only numeric columns
    numeric_df = dataframe.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame has no numeric columns to compute correlation.")
    
    # Calculate correlation matrix
    corr = numeric_df.corr(method=method)
    
    # Load environment variables and setup API
    load_dotenv()
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    client = genai.Client()
    
    #AI analysis gen
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
        You are a senior data analyst. Analyze this {method} correlation matrix and provide actionable business insights.

        Correlation Matrix:
        {corr}

        Please provide your analysis in the following structured format:

        1.  Strongest Relationships: Identify the 2-3 strongest positive and negative correlations (mention the features and correlation values).
        2.  Key Insight: For the strongest correlation, explain what this relationship might mean in a business context. Propose a hypothesis for why this relationship exists.
        3.  Unexpected Finding: Point out one correlation that is surprising or counter-intuitive and suggest a reason why it might be that way.
        4.  Recommendation: Suggest one next step for analysis (e.g., "Investigate the relationship between X and Y with a scatter plot to check for non-linear patterns" or "Explore if variable Z is a confounding factor in the relationship between X and Y").
        5.  When providing analysis be aware that it will be saved as a txt file and should not include any markdown or bolding
        """
    )
    
    #cleaning
    report_text = response.text
    
    #remove any potential markdown formatting
    report_text = report_text.replace('**', '').replace('*', '').replace('#', '')
    
    # Print with clean formatting
    print("=" * 70)
    print("CORRELATION ANALYSIS REPORT")
    print("=" * 70)
    print(report_text)
    print("=" * 70)

    return report_text

def data_quality_report(dataframe):
    """
    Generate a comprehensive data quality assessment with AI-powered insights including outlier detection
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        Input dataframe to analyze
    
    Returns:
    --------
    str: AI-generated data quality analysis report
    """
    
    # Input validation
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    
    # Calculate basic metrics
    missing_values = dataframe.isnull().sum()
    missing_percentage = (missing_values / len(dataframe)) * 100
    duplicate_rows = dataframe.duplicated().sum()
    data_types = dataframe.dtypes
    numeric_stats = dataframe.describe() if dataframe.select_dtypes(include=[np.number]).shape[1] > 0 else None
    
    # Outlier detection for numeric columns
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    outlier_report = {}
    
    for col in numeric_cols:
        # Remove NaN values for outlier calculation
        col_data = dataframe[col].dropna()
        
        if len(col_data) > 0:
            # IQR method for outliers
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Z-score method (for normally distributed data)
            z_scores = np.abs(stats.zscore(col_data))
            outliers_z = col_data[z_scores > 3]
            
            outlier_report[col] = {
                'iqr_outliers_count': len(outliers_iqr),
                'iqr_outliers_percentage': (len(outliers_iqr) / len(col_data)) * 100,
                'z_score_outliers_count': len(outliers_z),
                'z_score_outliers_percentage': (len(outliers_z) / len(col_data)) * 100,
                'lower_bound_iqr': lower_bound,
                'upper_bound_iqr': upper_bound
            }
    
    # Load environment variables and setup API
    load_dotenv()
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    client = genai.Client()
    
    # Generate AI analysis
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
        You are a senior data analyst. Analyze this data quality assessment and provide actionable insights.

        Dataset Shape: {dataframe.shape}
        Missing Values: {dict(missing_values)}
        Missing Percentage: {dict(missing_percentage.round(2))}
        Duplicate Rows: {duplicate_rows}
        Data Types: {dict(data_types)}
        Numeric Statistics: {numeric_stats.to_string() if numeric_stats is not None else 'No numeric columns'}
        
        Outlier Analysis:
        {pd.DataFrame(outlier_report).T.to_string() if outlier_report else 'No numeric columns for outlier analysis'}

        Please provide your analysis in the following structured format:

        1. Critical Issues: Identify the 3 most serious data quality problems including missing data, duplicates, and outliers
        2. Impact Assessment: Explain how each issue might affect analysis, modeling, and business decisions
        3. Cleaning Recommendations: Specific, actionable steps to address each data quality issue
        4. Priority Order: Which issues to fix first based on severity and business impact
        5. Outlier Context: For each variable with outliers, suggest whether they represent errors or legitimate extreme values
        6. When providing analysis be aware that it will be saved as a txt file and should not include any markdown or bolding
        """
    )
    
    # Clean and print the report
    report_text = response.text
    
    # Remove any potential markdown formatting
    report_text = report_text.replace('**', '').replace('*', '').replace('#', '')
    
    # Print with clean formatting
    print("=" * 70)
    print("DATA QUALITY ASSESSMENT REPORT")
    print("=" * 70)
    print(report_text)
    print("=" * 70)
    
    return report_text

