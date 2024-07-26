import joblib
#Data Analysis
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluating Algorithms
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# loading the csv data to a Pandas DataFrame
data = pd.read_csv("C:/Users/User/Desktop/TravelGuardian/Hypertension-risk-model-main.csv")

# EDA
print(data.head())
print(data.info())

#Check Duplicated Values
print("\nDuplicated data:")
print(data.duplicated().sum())

# checking for missing values
missing_data=data.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage=(missing_data.sum()/data.shape[0])*100
print(f"The total percentage of missing data is {round(total_percentage,2)}%")
missing_data = missing_data.to_frame(name='Total')  # Convert Series to DataFrame
missing_data['Percentage'] = (missing_data['Total'] / len(data)) * 100

# Create a bar plot to visualize the percentage of missing data by feature
plt.figure(figsize=(9, 6))
sns.set(style="whitegrid")
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data=missing_data)
plt.title("Percentage of Missing Data by Feature")
plt.xlabel("Features", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

# Drop missing values
data.dropna(axis=0,inplace=True)

# checking for missing values
missing_data=data.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage=(missing_data.sum()/data.shape[0])*100
print(f"The total percentage of missing data is {round(total_percentage,2)}%")

sns.countplot(x="Risk",data=data)
plt.show()
cases=data.Risk.value_counts()
print(f"There are {cases[0]} patients without risk of Hypertension and {cases[1]} patients with risk of Hypertension")

fig=plt.figure(figsize=(15,20))
ax =fig.gca()
data.hist(ax = ax)
plt.show()


# Seperating Categorical colums and numerical colums
cate_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
      cont_val.append(column)

print(cate_val)
print(cont_val)

sns.set_theme(context='poster')
plt.figure(figsize=(10,7))
plt.title('Age distribution based on Hypertension Risk', color="Black",fontsize=25)

sns.distplot(data[data['Risk'] == 0]['age'], label='No risk of Hypertension')
sns.distplot(data[data['Risk'] == 1]['age'], label = 'Risk of Hypertension')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Separate features and target variable
X = data.drop(['Risk'], axis=1)
y = data['Risk']

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=12)
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

# Drop low-scored features and unmeasurable features
X = data.drop(['male', 'currentSmoker','cigsPerDay', 'diabetes', 'glucose','Risk'], axis=1) 

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# Define the RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)

# Fit the classifier on training data
classifier.fit(X_train, y_train)

# Initial prediction and accuracy score
initial_predict = classifier.predict(X_test)
print(f'Initial accuracy score: {accuracy_score(y_test, initial_predict)}')

# Perform GridSearchCV for hyperparameter tuning
parameters = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [2, 8, 10],
    'bootstrap': [True],
    'max_samples': [0.1, 0.5, 0.75],
    'max_features': ['sqrt', 'log2']
}
grid = GridSearchCV(classifier, param_grid=parameters, n_jobs=-1, cv=5, verbose=2)
grid.fit(X_train, y_train)

# Retrieve the best model and predictions
best_rf_model = grid.best_estimator_
prediction = best_rf_model.predict(X_test)

# Print the best parameters and the accuracy score
print(f'Best accuracy score: {accuracy_score(y_test, prediction)}')
print(f'Best parameters: {grid.best_params_}')

# Print the evaluation metrics
accuracy = accuracy_score(y_test, prediction)
report = classification_report(y_test, prediction)
conf_matrix = confusion_matrix(y_test, prediction)

print(f'Best Random Forest Model: {best_rf_model}')
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Plot the confusion matrix
plt.figure(figsize=(10, 6))
plt.title('Confusion Matrix', fontsize=16)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
plt.xlabel('Predicted Value', fontsize=14)
plt.ylabel('Actual Value', fontsize=14)
plt.show()

# Save the model, scaler, and feature names
joblib.dump(best_rf_model, 'hypertension_model.pkl')
joblib.dump(scaler, 'hypertension_scaler.pkl')
hypertension_feature_names = X.columns
joblib.dump(hypertension_feature_names, 'hypertension_feature_names.pkl')

# from sklearn.svm import SVC
# # Define the SVM classifier
# svm = SVC()

# # Set up the parameter grid for hyperparameter tuning
# params_svm = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto']
# }

# # Use GridSearchCV to find the best hyperparameters with stratified K-fold cross-validation
# skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
# grid_svm = GridSearchCV(svm, params_svm, cv=skf, n_jobs=-1)
# grid_svm.fit(X_train, y_train)
# best_svm_model = grid_svm.best_estimator_

# # Perform cross-validation on the best SVM model
# cv_scores = cross_val_score(best_svm_model, X_train, y_train, cv=skf)
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Average cross-validation score: {cv_scores.mean()}")

# # Predict and evaluate the model 
# y_pred_svm = best_svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_svm)
# report = classification_report(y_test, y_pred_svm)
# conf_matrix = confusion_matrix(y_test, y_pred_svm)

# # Plot the confusion matrix
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix for Testing Data', fontsize=16)
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()

# # Print the evaluation metrics
# print(f'Best SVM Model: {best_svm_model}')
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# # Define the Decision Tree classifier
# dt = DecisionTreeClassifier(random_state=42)

# # Perform GridSearchCV for hyperparameter tuning
# parameters = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# grid_dt = GridSearchCV(dt, param_grid=parameters, cv=5, n_jobs=-1, verbose=2)
# grid_dt.fit(X_train, y_train)

# # Retrieve the best model and predictions
# best_dt_model = grid_dt.best_estimator_
# prediction_dt = best_dt_model.predict(X_test)

# # Print the best parameters and the accuracy score
# print(f'Best accuracy score: {accuracy_score(y_test, prediction_dt)}')
# print(f'Best parameters: {grid_dt.best_params_}')

# # Print the evaluation metrics
# accuracy_dt = accuracy_score(y_test, prediction_dt)
# report_dt = classification_report(y_test, prediction_dt)
# conf_matrix_dt = confusion_matrix(y_test, prediction_dt)

# print(f'Best Decision Tree Model: {best_dt_model}')
# print(f'Accuracy: {accuracy_dt}')
# print(f'Classification Report:\n{report_dt}')
# print(f'Confusion Matrix:\n{conf_matrix_dt}')

# # Plot the confusion matrix
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix (Decision Tree)', fontsize=16)
# sns.heatmap(conf_matrix_dt, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()

# # Save the model and feature names
# joblib.dump(best_dt_model, 'hypertension_dt_model.pkl')
# hypertension_feature_names_dt = X.columns
# joblib.dump(hypertension_feature_names_dt, 'hypertension_feature_names_dt.pkl')