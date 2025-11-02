import pandas as pd

# Our sample data
sales_data = pd.DataFrame({
    'Region': ['North', 'North', 'South', 'South', 'North', 'South'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2'],
    'Sales': [1000, 1500, 1200, 1800, 1100, 1600],
    'Units': [50, 75, 60, 90, 55, 80]
})

# Total sales by region and product
sales_by_region = pivot_df(sales_data, 
                          index='Region', 
                          columns='Product', 
                          values='Sales', 
                          aggfunc='sum')

# Average units sold by quarter and region
units_by_quarter = pivot_df(sales_data,
                           index='Quarter',
                           columns='Region', 
                           values='Units',
                           aggfunc='mean')

# Multiple aggregations with margins
summary = pivot_df(sales_data,
                  index='Region',
                  columns='Quarter',
                  values=['Sales', 'Units'],
                  aggfunc={'Sales': 'sum', 'Units': 'mean'},
                  margins=True)
