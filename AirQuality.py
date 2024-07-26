import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("C:/Users/User/Desktop/TravelGuardian/air_quality_health_impact_data.csv")

# EDA
print(df.head())
print(df.info())

# Check Duplicated Values
print("\nDuplicated data:")
print(df.duplicated().sum())

missing_data = df.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage = (missing_data.sum() / df.shape[0]) * 100
print(f"The total percentage of missing data is {round(total_percentage, 2)}%")

# Drop 'RecordID' column
df = df.drop(columns='RecordID')
print(df.head(2))

# Function to map HealthImpactClass to numerical value
def impact(x):
    if x == 0:
        return 100
    elif x == 1:
        return 75
    elif x == 2:
        return 50
    elif x == 3:
        return 25
    else:
        return 0

# Apply the function to HealthImpactClass
df['HealthImpactClass'] = df['HealthImpactClass'].apply(impact)
print(df.head(2))

# Visualize class distribution
df['HealthImpactClass'].value_counts().plot(kind='pie', autopct='%0.2f')
plt.show()

# Boxplot for features
plt.figure(figsize=(20, 10))
for i, col in enumerate(df.columns[0:13]):
    plt.subplot(3, 5, i + 1)
    sns.boxplot(df[col], orient='h')
    plt.title(f'{col}')
plt.show()

sns.set_theme(context='poster')
plt.figure(figsize=(10, 7))
plt.title('AQI distribution based on Health Impact Class', color="Black", fontsize=25)

sns.distplot(df[df['HealthImpactClass'] == 100]['AQI'], label='Very High')
sns.distplot(df[df['HealthImpactClass'] == 75]['AQI'], label='High')
sns.distplot(df[df['HealthImpactClass'] == 50]['AQI'], label='Medium')
sns.distplot(df[df['HealthImpactClass'] == 25]['AQI'], label='Low')
sns.distplot(df[df['HealthImpactClass'] == 0]['AQI'], label='Very Low')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Separate features and target variable and Drop unmeasurable features 
X = df.drop(['HealthImpactClass', 'HealthImpactScore', 'CardiovascularCases', 'HospitalAdmissions', 'RespiratoryCases'], axis=1)
y = df['HealthImpactClass']

# Apply SMOTE to oversample the minority classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=13)
X_selected = selector.fit_transform(X_resampled, y_resampled)

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

# Prepare the data for training
X_selected = pd.DataFrame(X_selected, columns=selected_features)
x_train, x_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42)

# # Train the RandomForestClassifier
# classifier = RandomForestClassifier()
# classifier.fit(x_train, y_train)
# predict = classifier.predict(x_test)
# print(f"Initial accuracy score: {accuracy_score(y_test, predict)}")

# # Perform GridSearchCV for hyperparameter tuning
# parameters = {
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [2, 8, 10],
#     'bootstrap': [True],
#     'max_samples': [0.1, 0.5, 0.75],
#     'max_features': ['sqrt', 'log2']
# }
# grid = GridSearchCV(classifier, param_grid=parameters, n_jobs=-1)
# grid.fit(x_train, y_train)
# # Retrieve the best model and predictions
# best_rf_model = grid.best_estimator_
# prediction = best_rf_model.predict(x_test)

# # Print the best parameters and the accuracy score
# print(f'Best accuracy score: {accuracy_score(y_test, prediction)}')
# print(f'Best parameters: {grid.best_params_}')

# # Print the evaluation metrics
# accuracy = accuracy_score(y_test, prediction)
# report = classification_report(y_test, prediction)
# conf_matrix = confusion_matrix(y_test, prediction)

# print(f'Best Random Forest Model: {best_rf_model}')
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# # Plot the confusion matrix
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix', fontsize=16)
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()

# # Save the model, scaler, and feature names
# joblib.dump(best_rf_model, 'airquality_model.pkl')
# joblib.dump(scaler, 'airquality_scaler.pkl')
# airquality_feature_names = X.columns
# print(X.columns)
# joblib.dump(airquality_feature_names, 'airquality_feature_names.pkl')

# Train a KNN model with GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(knn, params_knn, cv=6, n_jobs=-1)
grid_knn.fit(x_train, y_train)
best_knn_model = grid_knn.best_estimator_

# Predict and evaluate the model
y_pred_knn = best_knn_model.predict(x_test)
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
joblib.dump(best_knn_model, 'airquality_knn_model.pkl')
joblib.dump(scaler, 'airquality_scaler.pkl')
heart_feature_names = X.columns
print(X.columns)
joblib.dump(heart_feature_names, 'airquality_feature_names.pkl')

# Initialize and fit the initial Decision Tree
# classifier = DecisionTreeClassifier(random_state=42)
# classifier.fit(x_train, y_train)
# initial_predict = classifier.predict(x_test)
# print(f'Initial accuracy score: {accuracy_score(y_test, initial_predict)}')

# # Perform GridSearchCV for hyperparameter tuning
# parameters = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 10, 20],
#     'min_samples_leaf': [1, 5, 10]
# }
# grid = GridSearchCV(classifier, param_grid=parameters, n_jobs=-1, cv=5, verbose=2)
# grid.fit(x_train, y_train)

# # Retrieve the best model and predictions
# best_dt_model = grid.best_estimator_
# prediction = best_dt_model.predict(x_test)

# # Print the best parameters and the accuracy score
# print(f'Best accuracy score: {accuracy_score(y_test, prediction)}')
# print(f'Best parameters: {grid.best_params_}')

# # Print the evaluation metrics
# accuracy = accuracy_score(y_test, prediction)
# report = classification_report(y_test, prediction)
# conf_matrix = confusion_matrix(y_test, prediction)

# print(f'Best Decision Tree Model: {best_dt_model}')
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# # Plot the confusion matrix
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix', fontsize=16)
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()