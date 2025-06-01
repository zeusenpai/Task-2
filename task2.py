import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# loading the dataset
df = pd.read_csv('Titanic-Dataset.csv')

#cleaning the dataset/ processing
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

df.drop('Name', axis=1, inplace=True)   
df.drop('Ticket', axis=1, inplace=True)

df['Sex_Encoded'], _ = pd.factorize(df['Sex'])
df.drop('Sex', axis=1, inplace=True)
df.rename(columns={'Sex_Encoded': 'Sex'}, inplace=True)

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True, dtype=int)

numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

numerical_features_to_scale = [col for col in numerical_features if col in df.columns]

if numerical_features_to_scale:
    for col in numerical_features_to_scale:
        min_val = df[col].min()
        max_val = df[col].max()
        if (max_val - min_val) != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0

numerical_features_for_outliers = ['Age', 'SibSp', 'Parch', 'Fare']
numerical_features_for_outliers = [col for col in numerical_features_for_outliers if col in df.columns]
df_cleaned = df.copy()
def remove_outliers_iqr(dataframe, column_name):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]
    return df_cleaned

for col in numerical_features_for_outliers:
    if col in df_cleaned.columns:
        df_cleaned = remove_outliers_iqr(df_cleaned, col)



# Generate summary statistics (mean, median, std, etc.)
print(df_cleaned.describe())

# Create histograms and boxplots for numeric features
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
if 'PassengerId' in numeric_cols:
    numeric_cols.remove('PassengerId')
if 'Survived' in numeric_cols:
    numeric_cols.remove('Survived')

# Histograms using Plotly
print("\nGenerating Histograms using Plotly...")
for col in numeric_cols:
    fig = px.histogram(df_cleaned, x=col, marginal="box",
                       title=f'Histogram of {col}',
                       template="plotly_white")
    fig.show()

# Boxplots using Plotly
print("\nGenerating Boxplots using Plotly...")
for col in numeric_cols:
    fig = px.box(df_cleaned, y=col,
                 title=f'Boxplot of {col}',
                 template="plotly_white")
    fig.show()

# Use pairplot/correlation matrix for feature relationships
# Correlation Matrix
print("\nCorrelation Matrix:")
correlation_matrix = df_cleaned.corr()
print(correlation_matrix)

# Heatmap of the Correlation Matrix using Plotly
print("\nGenerating Heatmap of the Correlation Matrix using Plotly...")
fig = px.imshow(correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix of Features')
fig.show()

# Pairplot for selected numerical features using Plotly (scatter_matrix)
pairplot_features = ['Survived', 'Pclass', 'Age', 'Fare', 'Sex']
pairplot_features = [col for col in pairplot_features if col in df_cleaned.columns] # Ensure columns exist

if len(pairplot_features) > 1:
    print("\nGenerating Pairplot (for selected features: {}) using Plotly...".format(pairplot_features))
    fig = px.scatter_matrix(df_cleaned,
                            dimensions=pairplot_features,
                            color='Survived',
                            title='Pairplot of Selected Features by Survived Status',
                            template="plotly_white")
    fig.update_traces(diagonal_visible=False)
    fig.show()


