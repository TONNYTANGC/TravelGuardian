# Importing libraries
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss

df = pd.read_csv('C:/Users/User/Desktop/TravelGuardian/asthma_disease_data.csv')

# EDA
print(df.head())
print(df.info())

# Check duplicated values
print("\nDuplicated data:")
print(df.duplicated().sum())

missing_data = df.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage = (missing_data.sum() / df.shape[0]) * 100
print(f"The total percentage of missing data is {round(total_percentage, 2)}%")

df['Diagnosis'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')

# Adding titles and labels
plt.title('Value Counts of Target Variable', fontsize=16)
plt.xlabel('Target', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Display the plot
plt.show()

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# Dropping 'PatientID' and 'DoctorInCharge' as they are not useful for modeling
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

# Correlation heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Spliting
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=27)
X_selected = selector.fit_transform(X, y)

# Create a DataFrame to store features and their scores
selected_features = X.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]
feature_score_df = pd.DataFrame({'Features': selected_features, 'Scores': feature_scores}).sort_values(by='Scores', ascending=False)

# Plot feature scores
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Scores', y='Features', data=feature_score_df, palette='magma', dodge=False)
plt.title('Feature Score', fontsize=18)
plt.xlabel('Scores', fontsize=16)
plt.ylabel('Features', fontsize=16)
for container in ax.containers:
    ax.bar_label(container)
plt.show()

# Drop features with scores lower than 0.8 from the original DataFrame X
high_score_features = feature_score_df[feature_score_df['Scores'] >= 1.0]['Features']
X = X[high_score_features]
print(X.columns) 

# Apply SMOTE to oversample the minority classes
smote_tomek = SMOTETomek(random_state=42)
X, y = smote_tomek.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# from sklearn.linear_model import LogisticRegression
# log_reg = LogisticRegression()
# params_log_reg = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
# skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

# grid_log_reg = GridSearchCV(log_reg, params_log_reg,cv=skf, n_jobs=-1)
# grid_log_reg.fit(X_train, y_train)
# best_log_reg_model = grid_log_reg.best_estimator_

# cv_scores = cross_val_score(best_log_reg_model, X_train, y_train, cv=skf)
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Average cross-validation score: {cv_scores.mean()}")

# # Predict and evaluate the model
# y_pred_log_reg = best_log_reg_model.predict(X_test)
# print(f'Accuracy Score for {best_log_reg_model}:', accuracy_score(y_test, y_pred_log_reg))
# print(classification_report(y_test, y_pred_log_reg))

# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred_log_reg)
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix for New Testing Data', fontsize=16)
# sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()

# # Print the evaluation metrics
# accuracy = accuracy_score(y_test, y_pred_log_reg)
# report = classification_report(y_test, y_pred_log_reg)
# conf_matrix = confusion_matrix(y_test, y_pred_log_reg)

# print(f'Best Logistic Regression Model: {best_log_reg_model}')
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# Train a KNN model with GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# Define the KNN classifier
knn = KNeighborsClassifier()

# Set up the parameter grid for hyperparameter tuning
params_knn = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance']}

# Use GridSearchCV to find the best hyperparameters with stratified K-fold cross-validation
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
grid_knn = GridSearchCV(knn, params_knn, cv=skf, n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn_model = grid_knn.best_estimator_

# Perform cross-validation on the best KNN model
cv_scores = cross_val_score(best_knn_model, X_train, y_train, cv=skf)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Predict and evaluate the model 
y_pred_knn = best_knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn)
report = classification_report(y_test, y_pred_knn)
conf_matrix = confusion_matrix(y_test, y_pred_knn)

# Plot the confusion matrix
plt.figure(figsize=(10, 6))
plt.title('Confusion Matrix for New Testing Data', fontsize=16)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
plt.xlabel('Predicted Value', fontsize=14)
plt.ylabel('Actual Value', fontsize=14)
plt.show()

# Print the evaluation metrics
print(f'Best KNN Model: {best_knn_model}')
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Save the model, scaler, and feature names
joblib.dump(best_knn_model, 'asthma_knn_model.pkl')
joblib.dump(scaler, 'asthma_scaler.pkl')
heart_feature_names = X.columns
print(X.columns)
joblib.dump(heart_feature_names, 'asthma_feature_names.pkl')

# # Initialize Gaussian Naive Bayes model
# gnb = GaussianNB()

# # Define the parameter grid (for GaussianNB, there's typically not much to tune)
# params_gnb = {}

# # Set up cross-validation scheme
# skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

# # Perform grid search
# grid_gnb = GridSearchCV(gnb, params_gnb, cv=skf, n_jobs=-1)
# grid_gnb.fit(X_train, y_train)
# best_gnb_model = grid_gnb.best_estimator_

# # Perform cross-validation
# cv_scores = cross_val_score(best_gnb_model, X_train, y_train, cv=skf)
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Average cross-validation score: {cv_scores.mean()}")

# # Predict and evaluate the model
# y_pred_gnb = best_gnb_model.predict(X_test)
# print(f'Accuracy Score for {best_gnb_model}:', accuracy_score(y_test, y_pred_gnb))
# print(classification_report(y_test, y_pred_gnb))

# # Plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred_gnb)
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix for New Testing Data', fontsize=16)
# sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()

# # Print the evaluation metrics
# accuracy = accuracy_score(y_test, y_pred_gnb)
# report = classification_report(y_test, y_pred_gnb)
# conf_matrix = confusion_matrix(y_test, y_pred_gnb)

# print(f'Best Gaussian Naive Bayes Model: {best_gnb_model}')
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# print(f'Confusion Matrix:\n{conf_matrix}')