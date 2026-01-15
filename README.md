# EMS
A simple to use data analysis system to quickly provides insights, visualizations, and statistical summaries of your datasets. with minimal user coding required. <br>
It is designed to be easily imported into Python, EMS streamlines exploratory data analysis, allowing you to identify trends, correlations, and key metrics with minimal code. Perfect for data scientists, analysts, and developers who want fast, clear, and actionable data insights.
# Getting Started
### Instalation
- Clone this repo (or download as .zip file and extract)<br>
- Using the folder location on the users system run the following python code at the start of your project:
```python
import sys
sys.path.append(r"C:\Users\HughP01\EMS-Toolkit") # <-- Your file location in the brackets
from ems import *
```
### Installation Verification
After installing the module, verify everything is working correctly with this simple test:

```python
from ems import marco
marco()
```

If the installation was successful, you should see:
```python
polo
```

The project is now installed and you have access to all functions in the EMS Toolkit.

Note: To use the built in AI features please make sure that you have the necessary API keys saved on your system variabls. eg. GEMINI_API_KEY

For usage documentation please find our [Usage Guide](USAGE.md).
# Features
### Built in Artificial Intelligence tools
- Built in Gemini API
- Automatic Report Generation
- Data Analysis Pipeline Suggestions
  
### Data Quality & Cleaning
- Identify duplicate, null, and unique entries
- Remove duplicates and outliers
- Handle missing values
- Convert column data types
- Remove unnecessary columns

### Data Analysis & Statistics
- Generate comprehensive summary statistics
- Calculate correlation matrices
- Perform frequency counts and distribution analysis
- Identify and analyze outliers
- Pivot tables for grouping and aggregation (sums, averages, counts, etc.)
- Data reshaping

### Data Transformation
- Normalize and standardize data
- Apply log transformations
- Data scaling and normalization techniques

### Visualization
- **Statistical Charts**: Boxplots, dot plots, line plots
- **Comparative Analysis**: Stacked bar charts, correlation heatmaps
- **Geospatial**: Choropleth maps
- Distribution plots and frequency visualizations

### Automated Reporting
- **Correlation Reports**: Detailed correlation analysis using Gemini
- **Data Quality Reports**: Comprehensive data assessment and quality metrics

### Machine Learning Tools
- Check if a GPU is avaliable for machine learning (Note: EMS will default to CPU)

### Machine Learning Pipelines
- **Regression Models**:
- **Classification Models**:

### Statistical Analysis
- *T-tests*:

# To-Do
Filtering and subsetting rows.<br>
Sorting and ranking.<br>
Feature engineering (e.g., creating ratios, lags, rolling averages).<br>
Plotting Time series<br>
Interactive dashboards.<br>
#### Testing 
Hypothesis testing (, chi-square, ANOVA)
