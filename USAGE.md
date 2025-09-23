# How to use
## Core Functions
- summary(data)
- correlation(data,method="pearson")
- data_check(data,detail=False)
- find_outliers(data,show_rows=False, show_details=False)
- pivot_df(data,index=None, columns=None, values=None, aggfunc='mean', fill_value=None, margins=False, margins_name='All', dropna=True)
- unique(data, columns=None)
## Graphing
- correlation_chart(data, method='pearson', size=(10,8), cmap='coolwarm')
- analyze_dist(data, figsize=(15, 10))
- create_boxplot(data, x_col, y_col, title="Box Plot", x_label=None, y_label=None)
- box_chart(data, outliers_only=False)
- stacked_chart(data, category_col, value_col, stack_col, title="Stacked Bar Chart", x_label=None, y_label=None, legend_title=None)
## Report Generation
## Machine Learning
