import pandas as pd

#Load data (many thanks to JehielT for hosting sample data)
url = "https://raw.githubusercontent.com/JehielT/World-Happiness-Report/master/data.csv"
df = pd.read_csv(url)
df.head()
