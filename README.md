<h1>Understand data using statistics and visualizations</h1>
<h3>Tools used:</h3>
<ol>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Plotly</li>
</ol>
<h3>Description:</h3>
<p>
The Titanic data analysis and preprocessing involved loading the dataset, checking for nulls and data types, and handling missing values by imputing 'Age' with the median, dropping 'Cabin' due to excessive nulls, and imputing 'Embarked' with the mode. Categorical features like 'Name' and 'Ticket' were dropped, while 'Sex' was converted to numerical using pandas.factorize() and 'Embarked' was One-Hot Encoded using pandas.get_dummies(). Numerical features such as 'Pclass', 'Age', 'SibSp', 'Parch', and 'Fare' were then manually scaled to a 0-1 range using Min-Max normalization. Outliers were visualized with boxplots and subsequently removed using the IQR method. Finally, the data was understood through summary statistics, histograms, boxplots, correlation matrices, heatmaps, and pairplots, revealing that 'Sex', 'Pclass', and 'Fare' were the most influential factors for survival, with females and higher-class passengers having better rates, while the outlier removal significantly altered the distributions of 'SibSp' and 'Parch', and 'Age' showed a very weak linear correlation with survival after preprocessing.</p>
