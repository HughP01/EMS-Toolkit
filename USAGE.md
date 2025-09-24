# How to use
## Core Functions
- summary(data)
- correlation(data,method="pearson")
- data_check(data,detail=False)
- find_outliers(data,show_rows=False, show_details=False)
- pivot_df(data,index=None, columns=None, values=None, aggfunc='mean', fill_value=None, margins=False, margins_name='All', dropna=True)
- unique(data, columns=None)
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
## Machine Learning
