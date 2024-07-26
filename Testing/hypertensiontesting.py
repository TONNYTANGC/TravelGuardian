#Data Analysis
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

# Evaluating Algorithms
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE

# loading the csv data to a Pandas DataFrame
data = pd.read_csv("C:/Users/User/Desktop/TravelGuardian/Hypertension-risk-model-main.csv")
data.dropna(axis=0,inplace=True)
# Drop low-scored features and unmeasurable features to have better KMEAN Performance
X = data.drop(['male', 'cigsPerDay', 'diabetes', 'glucose','currentSmoker','Risk'], axis=1) 
y = data['Risk']

# Standardize the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Elbow Method for optimal number of clusters
inertia = []
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, marker='o', color='r')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.show()

# Apply KMeans clustering to create severity levels
kmeans = KMeans(n_clusters=3, random_state=42)
data['Severity'] = kmeans.fit_predict(X_scaled)
print(data['Severity'].value_counts())

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['age'], y=data['sysBP'], hue=data['Severity'], palette='viridis')
plt.title('KMeans Clusters')
plt.show()

# Invert severity values to map Severity 0 to 100 and accordingly
severity_max = data['Severity'].max()
data['Severity_Inverted'] = severity_max - data['Severity']

# Normalize the inverted severity values
severity_inverted_min = data['Severity_Inverted'].min()
severity_inverted_max = data['Severity_Inverted'].max()
data['Severity_Normalized'] = 100 * (data['Severity_Inverted'] - severity_inverted_min) / (severity_inverted_max - severity_inverted_min)

# Drop the intermediate 'Severity_Inverted' column
data.drop(columns=['Severity_Inverted'], inplace=True)

# Verify the normalization
print(data[['Severity', 'Severity_Normalized']]) 

# Drop low-scored features and unmeasurable features
# Separate features and target variable for classification and apply Standard Scaler
X = data.drop(['Severity', 'Severity_Normalized','male', 'cigsPerDay', 'diabetes', 'glucose','currentSmoker', 'Risk'], axis=1)
y = data['Severity_Normalized'] 


