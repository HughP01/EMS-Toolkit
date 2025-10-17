# EMS - Extra Modelling System
EMS - Extra Modelling System is a simple to use data analysis system to quickly provides insights, visualizations, and statistical summaries of your datasets. with minimal user coding required. <br>
It is designed to be easily imported into Python, EMS streamlines exploratory data analysis, allowing you to identify trends, correlations, and key metrics with minimal code. Perfect for data scientists, analysts, and developers who want fast, clear, and actionable data insights.
# Getting Started
For usage documentation please find our [Usage Guide](USAGE.md).
### Instalation
- Clone this repo (or download as .zip file and extract)<br>
- Using the folder location on the users system run the following python code at the start of your project:
```python
import sys
sys.path.append(r"C:\Users\HughP01\EMS-Toolkit\ems") # <-- Your file location in the brackets
from ems import *
```
- The project is now installed and you have access to all functions in the EMS Toolkit.


# Features

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
- Pivoting and data reshaping

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

# To-Do
Filtering and subsetting rows.<br>
Sorting and ranking.<br>
Grouping and aggregation (sums, averages, counts, etc.).<br>
reshaping (wide vs long format).<br>
Feature engineering (e.g., creating ratios, lags, rolling averages).<br>
gemini: data type report, dist. report,  <br>
Plotting Time series<br>
Interactive dashboards.<br>
Automated reporting (PDF, HTML, notebooks).
#### Testing 
Hypothesis testing (t-tests, chi-square, ANOVA)
Classification Models
