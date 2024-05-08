Loading Data:
python
Copy code
import pandas as pd

# Load data from CSV file
df = pd.read_csv('data.csv')
Viewing Data:
python
Copy code
# Display the first few rows of the dataframe
print(df.head())

# Display the last few rows of the dataframe
print(df.tail())

# Get a concise summary of the dataframe
print(df.info())
Handling Missing Values:
python
Copy code
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)
Handling Duplicates:
python
Copy code
# Check for duplicate rows
print(df.duplicated().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)
Handling Outliers:
python
Copy code
# Visualize boxplot for outlier detection
import seaborn as sns
sns.boxplot(x=df['column_name'])

# Remove outliers using IQR method
Q1 = df['column_name'].quantile(0.25)
Q3 = df['column_name'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['column_name'] < (Q1 - 1.5 * IQR)) | (df['column_name'] > (Q3 + 1.5 * IQR)))]
Data Transformation:
python
Copy code
# Apply a function to transform data
df['column_name'] = df['column_name'].apply(lambda x: x**2)

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['categorical_column'])
Data Aggregation:
python
Copy code
# Group data by a categorical column and calculate mean
df_grouped = df.groupby('category_column').mean()

# Pivot table for aggregation
pivot_table = pd.pivot_table(df, index='column1', columns='column2', values='value', aggfunc=np.mean)
Data Merging:
python
Copy code
# Merge two dataframes based on a common column
merged_df = pd.merge(df1, df2, on='common_column')
Data Reshaping:
python
Copy code
# Transpose dataframe
transposed_df = df.T

# Melt dataframe
melted_df = pd.melt(df, id_vars=['id'], value_vars=['var1', 'var2'], var_name='variable', value_name='value')
Data Sampling:
python
Copy code
# Randomly sample data
sampled_df = df.sample(n=100, random_state=42)  # n is the number of samples