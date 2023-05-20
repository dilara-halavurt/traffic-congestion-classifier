import pandas as pd

# Read the files into two dataframes.
df1 = pd.read_csv('results.csv', delimiter=';')
df2 = pd.read_csv('your_file.csv')

print(df1.columns)
print(df2.columns)

# Merge the two dataframes, using _ID column as key
df3 = pd.merge(df1, df2, on='Name')
df3.set_index('Name', inplace=True)

# Write it to a new CSV file
df3.to_csv('CSV3.csv')
