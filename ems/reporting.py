from google import genai
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

genai.configure(api_key=os.getenv('GEMINI_API_KEY')) #Have API key saved as GEMINI_API_KEY in sys variables


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



corr_report(df)
