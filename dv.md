Histogram:
python
Copy code
import matplotlib.pyplot as plt

# Plot histogram
plt.hist(df['column_name'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Histogram of Column')
plt.show()
Scatter Plot:
python
Copy code
# Plot scatter plot
plt.scatter(df['x_column'], df['y_column'], color='blue', alpha=0.5)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Scatter Plot')
plt.show()
Box Plot:
python
Copy code
import seaborn as sns

# Plot box plot
sns.boxplot(x='category_column', y='numeric_column', data=df)
plt.xlabel('Category')
plt.ylabel('Numeric Column')
plt.title('Box Plot')
plt.show()
Bar Plot:
python
Copy code
# Plot bar plot
sns.barplot(x='category_column', y='numeric_column', data=df, estimator=np.mean)
plt.xlabel('Category')
plt.ylabel('Mean Numeric Column')
plt.title('Bar Plot')
plt.show()
Heatmap:
python
Copy code
# Plot heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()
Line Plot:
python
Copy code
# Plot line plot
plt.plot(df['x_column'], df['y_column'], marker='o', linestyle='-')
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Line Plot')
plt.show()
Violin Plot:
python
Copy code
# Plot violin plot
sns.violinplot(x='category_column', y='numeric_column', data=df)
plt.xlabel('Category')
plt.ylabel('Numeric Column')
plt.title('Violin Plot')
plt.show()