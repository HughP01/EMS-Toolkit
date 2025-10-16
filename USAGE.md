# Data Analysis Toolkit Documentation
## Core Functions

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

## Preprocessing

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

## Graphing
- correlation_chart(data, method='pearson', size=(10,8), cmap='coolwarm')
- analyze_dist(data, figsize=(15, 10))
- create_boxplot(data, x_col, y_col, title="Box Plot", x_label=None, y_label=None)
- box_chart(data, outliers_only=False)
- stacked_chart(data, category_col, value_col, stack_col, title="Stacked Bar Chart", x_label=None, y_label=None, legend_title=None)
- dot_plot(data, x_col, y_col, hue_col=None, title="Scatter Plot",x_label=None, y_label=None, figsize=(10, 6), alpha=0.7, palette='viridis', save_path=None, reg_line=False,ci=95, reg_color='red', reg_alpha=0.5,reg_width=2)
- world_chart(data, countries, column, locationmode='country names', color_scale='Viridis', title=None, width=1000, height=600)
- show_world_chart(data,country,column)
- lineplot(data, x_col, y_col, hue_col=None, title="Line Chart",  x_label=None, y_label=None, figsize=(10, 6), alpha=0.8, palette='viridis', save_path=None, ci=95, style_col=None, markers=False, linewidth=2.5,err_style='band', err_alpha=0.3)

## Report Generation

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
## Machine Learning
