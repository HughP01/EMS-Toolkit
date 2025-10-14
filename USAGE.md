# Data Analysis Toolkit Documentation
## Core Functions

### generate_dataframe_summary

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

### summary

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

### correlation

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

### data_check

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

### find_outliers

**Description:**  
Identifies outliers in numeric columns using the Interquartile Range (IQR) method.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame to analyze
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

### pivot_df

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

### unique

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
- handle_missing_values(data, strategy='avg', columns=None)
- remove_duplicates(data, subset=None)
- drop_columns(data, columns_to_drop)
- standardize(data, columns=None)
- normalize(data, columns=None)
- log_transform(data, columns, add_one=True)
- remove_outliers_iqr(data, columns=None, threshold=1)
- convert(data, columns, target_dtype, errors='ignore', inplace=True)
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
- corr_report(dataframe, method='pearson')
- data_quality_report(data)
## Machine Learning
