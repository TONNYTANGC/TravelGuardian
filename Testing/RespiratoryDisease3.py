import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/User/Desktop/TravelGuardian/cancer patient data sets.csv")

# Display the first few rows of the dataframe
print(df.head())

# Display information about the dataframe
print(df.info())

# Display summary statistics of the dataframe
print(df.describe())

# Display the count of missing values for each column
print(df.isnull().sum())

# Plot using seaborn
sns.barplot(y='Alcohol use', x='Swallowing Difficulty', data=df)
plt.show()  # Show the plot

sns.barplot(y='Weight Loss', x='Smoking', data=df)
plt.show()  # Show the plot

plt.scatter(df['Smoking'], df['Weight Loss'])
plt.xlabel('Smoking')
plt.ylabel('Weight Loss')
plt.show()  # Show the plot

sns.barplot(y='Level', x='Smoking', data=df)
plt.show()  # Show the plot

# Pie chart
labels = ['1', '2', '3', '4', '5', '6', '7', '8']
colors = ['blue', 'yellow', 'green', 'orange', 'red', 'pink', 'brown', 'grey']
plt.pie(df['Smoking'].value_counts(), labels=labels, colors=colors)
plt.show()  # Show the plot

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Level'] = encoder.fit_transform(df['Level'])
df = df.drop('Patient Id', axis=1)
X = df.drop('Level', axis=1)
y = df['Level']

# Compute the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Set the correlation threshold
threshold = 0.1

# Identify features with correlation below the threshold with respect to 'Level'
low_corr_features = corr_matrix.index[abs(corr_matrix["Level"]) < threshold].tolist()

# Drop these low correlation features from the dataframe
df = df.drop(columns=low_corr_features)
df = df.drop(columns=['OccuPational Hazards', 'Genetic Risk'])
# Display the updated dataframe
print(df.info())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import necessary libraries for model evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost model
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.1,
    'eval_metric': 'merror'
}

num_rounds = 10
model = xgb.train(params, dtrain, num_rounds)

y_pred = model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# AdaBoost model
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=10, learning_rate=1.0, algorithm='SAMME.R')
abc.fit(X_train, y_train)
y_pred1 = abc.predict(X_test)
print(classification_report(y_test, y_pred1))

# RandomForest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini')
rf.fit(X_train, y_train)
y_pred2 = rf.predict(X_test)
print(classification_report(y_test, y_pred2))

# KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred3 = knn.predict(X_test)
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred3))
print("KNN Accuracy:", accuracy_score(y_test, y_pred3))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred3))