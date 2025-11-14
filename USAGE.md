# Data Analysis Toolkit Documentation
# Core Functions

### generate_dataframe_summary()

**Description:**  
Generates a comprehensive statistical summary of a pandas DataFrame as a structured dictionary.

**Parameters:**
- `data` (pd.DataFrame): The DataFrame to analyze
- `sample_size` (int): Number of sample rows to include in summary (default: 5)

**Returns:**
  - `dict`: Nested dictionary containing:
  - Basic information (shape, dimensions, memory usage)
  - Column details (names, dtypes, null counts, null percentages)
  - Data type distribution
  - Numeric column statistics (if present)
  - Categorical column statistics (if present)
  - Duplicate row information
  - Sample data records


---

### summary()

**Description:**  
Prints a formatted, human-readable summary of the DataFrame to console output.

**Parameters:**
- `data` (pd.DataFrame): The DataFrame to summarize
- `sample_size` (int): Number of sample rows to display

**Output:**
Formatted console output including:
- Basic DataFrame information
- Data type distribution
- Null values summary
- Numeric columns statistics (count, mean, min, max, std)
- Categorical columns statistics (unique values, most frequent values)
- Duplicate rows information
- Sample data preview

---

### correlation()

**Description:**  
Computes correlation matrix for numeric columns in the DataFrame.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `method` (str): Correlation calculation method - 'pearson', 'kendall', or 'spearman' (default: 'pearson')

**Returns:**
- `pd.DataFrame`: Correlation matrix with same index and columns as numeric columns

**Raises:**
- `TypeError`: If input is not a pandas DataFrame
- `ValueError`: If DataFrame contains no numeric columns

---

### data_check()

**Description:**  
Performs data quality checks for missing values and duplicate rows.

**Parameters:**
- `data` (pd.DataFrame): The DataFrame to check
- `detail` (bool): If True, provides detailed information including specific row indices and duplicate row content

**Output:**
Console output showing:
- Total missing values count with warning if > 0
- Total duplicate rows count with warning if > 0
- Columns containing missing values
- When `detail=True`: Specific row indices with missing values and duplicate row content

---

### find_outliers()

**Description:**  
Identifies outliers in numeric columns using the Interquartile Range (IQR) method.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame to analyze
- `show_rows` (bool): If True, displays actual outlier values and their indices
- `show_details` (bool): If True, shows detailed statistical information including quartiles and fences

**Output:**
Console report showing for each numeric column with outliers:
- Outlier count
- Statistical details (Q1, Q3, IQR, fences) when `show_details=True`
- Outlier values and indices when `show_rows=True`
- Preview message when more than 5 outliers exist

**Note:** Only displays columns that actually contain outliers

---

### pivot_df()

**Description:**  
Creates a pivot table from DataFrame with extensive customization options.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame to pivot
- `index` (str or list): Column(s) to use as index (rows) in pivot table
- `columns` (str or list): Column(s) to use as columns in pivot table (required)
- `values` (str or list): Column(s) to aggregate in pivot table
- `aggfunc` (function, str, list, or dict): Aggregation function(s) (default: 'mean')
- `fill_value` (scalar): Value to replace missing values with
- `margins` (bool): Add row/column margins (subtotals) (default: False)
- `margins_name` (str): Name for margin rows/columns (default: 'All')
- `dropna` (bool): Exclude columns with all NaN values (default: True)

**Returns:**
- `pd.DataFrame`: Pivoted DataFrame

**Raises:**
- `TypeError`: If input is not a pandas DataFrame
- `ValueError`: If required parameters missing or invalid

**Note:** Automatically selects first numeric column for values if not specified

---

### unique()

**Description:**  
Analyzes and displays unique values and their frequencies from various data structures.

**Parameters:**
- `data` (pd.DataFrame, list, or array-like): Input data to analyze
- `columns` (str, list, or None): Column name(s) to analyze. For DataFrames: if None, uses all columns. For non-DataFrame input: parameter is ignored.

**Output:**
Console output showing for each column or data series:
- All unique values with their frequency counts
- Total count of unique values
- Values sorted by frequency (most common first)

# Preprocessing

### handle_missing_values()

**Description:**  
Handles missing values in a pandas DataFrame using specified strategies.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `strategy` (str): Handling method - 'avg' (mean/mode imputation) or 'drop' (row removal)
- `columns` (list): Specific columns to process. If None, applies to all columns

**Returns:**
- `pd.DataFrame`: DataFrame with missing values handled

**Raises:**
- `ValueError`: If specified columns not found or invalid strategy provided

**Strategy Details:**
- 'avg': Numeric columns filled with mean, categorical columns filled with mode
- 'drop': Rows with missing values in specified columns are removed

---

### remove_duplicates()

**Description:**  
Removes duplicate rows from DataFrame.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `subset` (list): Columns to consider for duplicate identification. If None, uses all columns

**Returns:**
- `pd.DataFrame`: DataFrame with duplicates removed (keeps first occurrence)

---

### drop_columns()

**Description:**  
Removes specified columns from DataFrame.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns_to_drop` (list): Column names to remove

**Returns:**
- `pd.DataFrame`: DataFrame with specified columns removed

**Note:** Ignores non-existent columns without raising errors

---

### standardize()

**Description:**  
Applies z-score normalization to numeric columns (mean=0, std=1).

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns` (list): Specific numeric columns to standardize. If None, processes all numeric columns

**Returns:**
- `pd.DataFrame`: DataFrame with standardized columns

---

### normalize()

**Description:**  
Applies min-max normalization to numeric columns (scales to 0-1 range).

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns` (list): Specific numeric columns to normalize. If None, processes all numeric columns

**Returns:**
- `pd.DataFrame`: DataFrame with normalized columns

---

### log_transform()

**Description:**  
Applies logarithmic transformation to specified columns.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns` (list): Columns to transform
- `add_one` (bool): If True, uses log1p to handle zeros by adding 1 before transformation

**Returns:**
- `pd.DataFrame`: DataFrame with log-transformed columns

---

### remove_outliers_iqr()

**Description:**  
Removes outliers using Interquartile Range method.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns` (list): Specific numeric columns to process. If None, processes all numeric columns
- `threshold` (float): IQR multiplier for outlier detection (default: 1.5)

**Returns:**
- `pd.DataFrame`: DataFrame with outliers removed

**Method:**
- Lower bound: Q1 - threshold × IQR
- Upper bound: Q3 + threshold × IQR

---

### winsorize()

**Description:**  
Winsorizes outliers by capping at specified percentiles.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns` (list): Specific columns to winsorize. If None, processes all numeric columns
- `limits` (list): Lower and upper percentile limits [lower_limit, upper_limit] (default: [0.05, 0.05])

**Note:** Function implementation pending

---

### convert()

**Description:**  
Converts specified columns to target data types with comprehensive error handling.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `columns` (str or list): Column name(s) to convert
- `target_dtype` (str): Target data type: 'string', 'float', 'int', 'object', 'category', 'datetime'
- `errors` (str): Error handling: 'raise', 'coerce', 'ignore' (default: 'ignore')
- `inplace` (bool): If True, modifies original DataFrame (default: True)

**Returns:**
- `pd.DataFrame`: DataFrame with converted columns

**Supported Data Types:**
- Numeric: 'float', 'int'
- Text: 'string', 'object'
- Special: 'category', 'datetime'

**Raises:**
- `ValueError`: If unsupported target_dtype provided

# Graphing
### correlation_chart()

**Description:**
Generates a correlation heatmap from numeric columns in a pandas DataFrame.

**Parameters:**
- `data` (pd.DataFrame): The data to analyze
- `method` (str): Correlation method - 'pearson', 'kendall', or 'spearman' (default: 'pearson')
- `size` (tuple): Figure size for the plot (default: (10,8))
- `cmap` (str): Seaborn colormap (default: 'coolwarm')

**Returns:**
- matplotlib.figure.Figure: The heatmap figure

---

### analyze_dist()

**Description:**
Analyzes frequency counts and distributions for all columns in a DataFrame, displaying both numeric and categorical variables.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `figsize` (tuple): Figure size for the plots (default: (15,10))

---

### create_boxplot()

**Description:**
Creates a box plot comparing a categorical variable against a numerical variable.

**Parameters:**
- `data` (pd.DataFrame): Pandas DataFrame containing the data
- `x_col` (str): Column name for the x-axis (categorical variable)
- `y_col` (str): Column name for the y-axis (numerical variable)
- `title` (str): Title of the plot (default: "Box Plot")
- `x_label` (str): Label for the x-axis (uses x_col if None)
- `y_label` (str): Label for the y-axis (uses y_col if None)

---

### box_chart()

**Description:**
Creates box plots for all numeric columns in the DataFrame, with optional filtering for outliers.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame
- `outliers_only` (bool): Only show columns that contain outliers (default: False)

---

### stacked_chart()

**Description:**
Creates a stacked bar chart showing composition of categories using Pandas and Seaborn styling.

**Parameters:**
- `data` (pd.DataFrame): Pandas DataFrame containing the data
- `category_col` (str): Column name for the main categories (x-axis)
- `value_col` (str): Column name for the values (y-axis)
- `stack_col` (str): Column name for the stacking categories
- `title` (str): Title of the plot (default: "Stacked Bar Chart")
- `x_label` (str): Label for the x-axis (uses category_col if None)
- `y_label` (str): Label for the y-axis (uses value_col if None)
- `legend_title` (str): Title for the legend (uses stack_col if None)


### dot_plot()

**Description:**
Create a scatter plot using Seaborn with optional regression line and confidence intervals.

**Parameters:**
- `data` (pandas DataFrame): The DataFrame containing the data to plot
- `x_col` (str): Column name for the x-axis
- `y_col` (str): Column name for the y-axis
- `hue_col` (str, optional): Column name for color encoding (categorical variable)
- `title` (str, optional): Title of the plot (default: "Scatter Plot")
- `x_label` (str, optional): Label for x-axis (default: uses x_col)
- `y_label` (str, optional): Label for y-axis (default: uses y_col)
- `figsize` (tuple, optional): Figure size (width, height) in inches (default: (10, 6))
- `alpha` (float, optional): Transparency of points (default: 0.7)
- `palette` (str, optional): Color palette name (default: 'viridis')
- `save_path` (str, optional): Path to save the plot (e.g., 'plot.png')
- `reg_line` (bool, optional): Whether to add a regression line (default: False)
- `ci` (int or None, optional): Confidence interval size (e.g., 95 for 95% CI)
- `reg_color` (str, optional): Color of the regression line (default: 'red')
- `reg_alpha` (float, optional): Transparency of confidence interval (default: 0.5)
- `reg_width` (float, optional): Width of the regression line (default: 2)

**Returns:**
- matplotlib.axes.Axes: The axes object with the plot

---

### world_chart()

**Description:**
Create a choropleth world map with customization options using Plotly.

**Parameters:**
- `data` (pandas DataFrame): The DataFrame containing the data to plot
- `countries` (str): Column name containing country names or codes
- `column` (str): Column name for the color values
- `locationmode` (str, optional): Location mode - 'country names', 'ISO-3', or 'USA-states' (default: 'country names')
- `color_scale` (str, optional): Plotly color scale ('Viridis', 'Plasma', 'Reds', etc.) (default: 'Viridis')
- `title` (str, optional): Title of the plot
- `width` (int, optional): Figure width in pixels (default: 1000)
- `height` (int, optional): Figure height in pixels (default: 600)

**Returns:**
- plotly.graph_objects.Figure: The Plotly figure object

---

### show_world_chart()

**Description:**
Generates and displays the choropleth chart.

**Parameters:**
- `data` (pandas DataFrame): The DataFrame containing the data to plot
- `country` (str): Column name containing country names or codes
- `column` (str): Column name for the color values

---

### lineplot()

**Description:**
Create a line chart using Seaborn with optional confidence intervals, multiple hues, and styling options.

**Parameters:**
- `data` (pandas DataFrame): The DataFrame containing the data to plot
- `x_col` (str): Column name for the x-axis (should be numeric or datetime)
- `y_col` (str): Column name for the y-axis
- `hue_col` (str, optional): Column name for color encoding (categorical variable)
- `title` (str, optional): Title of the plot (default: "Line Chart")
- `x_label` (str, optional): Label for x-axis (default: uses x_col)
- `y_label` (str, optional): Label for y-axis (default: uses y_col)
- `figsize` (tuple, optional): Figure size (width, height) in inches (default: (10, 6))
- `alpha` (float, optional): Transparency of lines (default: 0.8)
- `palette` (str, optional): Color palette name (default: 'viridis')
- `save_path` (str, optional): Path to save the plot (e.g., 'plot.png')
- `ci` (int, None, or 'sd', optional): Confidence interval size or standard deviation
- `style_col` (str, optional): Column name for line style encoding
- `markers` (bool, optional): Whether to show markers on data points (default: False)
- `linewidth` (float, optional): Width of the lines (default: 2.5)
- `err_style` (str, optional): Style of error representation: 'band' or 'bars' (default: 'band')
- `err_alpha` (float, optional): Transparency of error region (default: 0.3)

**Returns:**
- matplotlib.axes.Axes: The axes object with the plot

# Report Generation

### corr_report()

**Description:**  
Generates an AI-powered correlation analysis report with business insights using Google's Gemini API.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame containing numeric columns for correlation analysis
- `method` (str): Correlation calculation method - 'pearson', 'kendall', or 'spearman' (default: 'pearson')

**Returns:**
- `str`: AI-generated correlation analysis report in plain text format

**Raises:**
- `TypeError`: If input is not a pandas DataFrame
- `ValueError`: If DataFrame contains no numeric columns or GEMINI_API_KEY is missing

**Report Structure:**
- Strongest positive and negative correlations
- Key business insights for strongest relationships
- Unexpected findings with potential explanations
- Actionable recommendations for further analysis

**Dependencies:**
- Requires GEMINI_API_KEY in environment variables
- Uses Google Gemini 2.5 Flash model for AI analysis

**Output:**
Formatted console display with cleaned text suitable for .txt file export

---

### data_quality_report()

**Description:**  
Generates a comprehensive AI-powered data quality assessment report including outlier detection and cleaning recommendations.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame to analyze for data quality issues

**Returns:**
- `str`: AI-generated data quality assessment report in plain text format

**Raises:**
- `TypeError`: If input is not a pandas DataFrame
- `ValueError`: If DataFrame is empty or GEMINI_API_KEY is missing

**Analysis Includes:**
- Missing values statistics and percentages
- Duplicate row count
- Data type distribution
- Descriptive statistics for numeric columns
- Outlier detection using IQR and Z-score methods
- Outlier counts and percentages per numeric column

**Report Structure:**
- Critical data quality issues identification
- Impact assessment on analysis and business decisions
- Specific cleaning recommendations
- Priority order for issue resolution
- Outlier context analysis (errors vs legitimate extremes)

**Dependencies:**
- Requires GEMINI_API_KEY in environment variables
- Uses Google Gemini 2.5 Flash model for AI analysis
- Uses scipy.stats for Z-score calculations

**Output:**
Formatted console display with cleaned text suitable for .txt file export


# Machine Learning

### gpu_check()

**Description:**  
Performs comprehensive GPU detection and verification for PyTorch and TensorFlow frameworks.

**Parameters:**  
None

**Returns:**  
- `bool`: True if GPU is available in either framework, False otherwise

**Output:**  
Detailed console report including:
- PyTorch GPU detection status
- TensorFlow GPU detection status
- CUDA version information for both frameworks
- TensorFlow device name
- Final GPU availability summary

---

### _get_model_from_string()

**Description:**  
Internal helper function that instantiates machine learning models from string identifiers for both regression and classification tasks.

**Parameters:**
- `model_str` (str): Model identifier string
- `model_type` (str): Type of model - "regression" or "classification"
- `use` (str): Computational hardware - "CPU" or "GPU"
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `sklearn estimator`: Instantiated model object

**Supported Regression Models:**
- "linear": LinearRegression
- "randomforest": RandomForestRegressor
- "xgboost": XGBRegressor (with GPU support when available)
- "svr": SVR
- "decisiontree": DecisionTreeRegressor
- "gradientboosting": GradientBoostingRegressor

**Supported Classification Models:**
- "logistic": LogisticRegression
- "randomforest": RandomForestClassifier
- "xgboost": XGBClassifier (with GPU support when available)
- "svc": SVC
- "decisiontree": DecisionTreeClassifier
- "gradientboosting": GradientBoostingClassifier

**Raises:**
- `ValueError`: If unsupported model type or model string provided

---

### Regr()

**Description:**  
End-to-end regression model pipeline with automated preprocessing, training, and evaluation.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame containing features and target variable
- `model` (sklearn estimator or str): Regression model or model string identifier
- `use` (str): Computational hardware - "CPU" or "GPU" (default: "CPU")
- `target` (str): Name of target variable column (required)
- `test_size` (float): Proportion of data for testing (default: 0.2)
- `random_state` (int): Random seed for reproducibility (default: 42)
- `preprocess` (bool): Enable automated preprocessing (default: True)
- `scale_features` (bool): Enable feature scaling (default: True)
- `verbose` (bool): Enable detailed progress output (default: True)

**Returns:**
- `dict`: Comprehensive results dictionary containing:
  - Trained model object
  - Preprocessing objects (imputers, scalers, encoders)
  - Training and test metrics (MAE, MSE, RMSE, R²)
  - Feature names
  - Training and test datasets
  - Predictions
  - Model type identifier

**Preprocessing Pipeline:**
- Median imputation for numeric missing values
- "Missing" value imputation for categorical variables
- Label encoding for categorical features
- Standard scaling for numeric features (optional)

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

---

### Classif()

**Description:**  
End-to-end classification model pipeline with automated preprocessing, training, and evaluation.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame containing features and target variable
- `model` (sklearn estimator or str): Classification model or model string identifier
- `use` (str): Computational hardware - "CPU" or "GPU" (default: "CPU")
- `target` (str): Name of target variable column (required)
- `test_size` (float): Proportion of data for testing (default: 0.2)
- `random_state` (int): Random seed for reproducibility (default: 42)
- `preprocess` (bool): Enable automated preprocessing (default: True)
- `scale_features` (bool): Enable feature scaling (default: True)
- `verbose` (bool): Enable detailed progress output (default: True)

**Returns:**
- `dict`: Comprehensive results dictionary containing:
  - Trained model object
  - Preprocessing objects (imputers, scalers, encoders)
  - Training and test metrics (Accuracy, Precision, Recall, F1-Score)
  - Feature names
  - Training and test datasets
  - Predictions
  - Model type identifier
  - Class labels

**Preprocessing Pipeline:**
- Median imputation for numeric missing values
- "Missing" value imputation for categorical variables
- Label encoding for categorical features
- Standard scaling for numeric features (optional)
- Stratified train-test split for balanced class distribution

**Evaluation Metrics:**
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)
- Detailed classification report
- Confusion matrix
