import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV

# Ignore warnings
warnings.filterwarnings('ignore')

# Load and preprocess the dataset
data = pd.read_csv("C:/Users/User/Desktop/TravelGuardian/heart.csv")
data.rename(columns={
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest_Pain',
    'trestbps': 'Resting_BP',
    'chol': 'Cholestrol',
    'fbs': 'Fasting_Blood_Sugar',
    'restecg': 'Resting_Electrocardiographic',
    'thalach': 'Max_Heart_Rate',
    'exang': 'Exercise_Induced_Angina',
    'oldpeak': 'Old_Peak',
    'slope': 'Slope',
    'ca': 'No_Major_Vessels',
    'thal': 'Thal',
    'target': 'Target'
}, inplace=True)

# EDA
print(data.head())
print(data.info())

missing_data=data.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage=(missing_data.sum()/data.shape[0])*100
print(f"The total percentage of missing data is {round(total_percentage,2)}%")

#Plot Target Values 
data['Target'].value_counts().plot(kind="bar", color=["green","blue"]) 
plt.show()

#Plot Heart Disease in function of Age and Max Heart Rate
plt.figure(figsize=(10,6))

plt.scatter(data.Age[data.Target==1], data.Max_Heart_Rate[data.Target==1],c="red")
plt.scatter(data.Age[data.Target==0], data.Max_Heart_Rate[data.Target==0],c="green")

plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Heart Rate")
plt.legend(["Disease","No disease"])
plt.show()

#Plot Checking Distribution of Numerical Features
plt.figure(figsize=(10,6))

num_features = ['Age', 'Resting_BP', 'Cholestrol','Max_Heart_Rate', 'Old_Peak']

# Plot data on each subplot
for i, column in enumerate(num_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(data[column], color='blue', kde=True, bins=30)

    plt.title(f'{column} Distribution')
    plt.xlabel(f'{column}')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plot Checking Distribution of Categorical Features
plt.figure(figsize=(12,10))

catag = ['Chest_Pain', 'Fasting_Blood_Sugar', 'Resting_Electrocardiographic','Exercise_Induced_Angina', 'Slope', 'No_Major_Vessels', 'Thal']

for i, column in enumerate(catag):
    plt.subplot(3,3,i+1)
    sns.countplot(x = data[column], data = data, palette = 'coolwarm', hue = data[column])
    plt.title(f'Distribution for {column.upper()} ')
    plt.xlabel(f'{column}')
    plt.ylabel('Frequency')
    plt.legend().remove()

plt.tight_layout()
plt.show()



# Separate features and target variable
X = data.drop(['Target'], axis=1)
y = data['Target']

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=13)
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

# Drop low-scored features and unmeasurable/unnecessary features
X = data.drop(['Resting_BP', 'Resting_Electrocardiographic', 'Cholestrol', 'Fasting_Blood_Sugar','Target'], axis=1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# # Train a KNN model with GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# # Define the KNN classifier
# knn = KNeighborsClassifier()

# # Set up the parameter grid for hyperparameter tuning
# params_knn = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance']}

# # Use GridSearchCV to find the best hyperparameters with stratified K-fold cross-validation
# skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
# grid_knn = GridSearchCV(knn, params_knn, cv=skf, n_jobs=-1)
# grid_knn.fit(X_train, y_train)
# best_knn_model = grid_knn.best_estimator_

# # Perform cross-validation on the best KNN model
# cv_scores = cross_val_score(best_knn_model, X_train, y_train, cv=skf)
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Average cross-validation score: {cv_scores.mean()}")

# # Predict and evaluate the model 
# y_pred_knn = best_knn_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_knn)
# report = classification_report(y_test, y_pred_knn)
# conf_matrix = confusion_matrix(y_test, y_pred_knn)

# # Plot the confusion matrix
# plt.figure(figsize=(10, 6))
# plt.title('Confusion Matrix for New Testing Data', fontsize=16)
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
# plt.xlabel('Predicted Value', fontsize=14)
# plt.ylabel('Actual Value', fontsize=14)
# plt.show()

# # Print the evaluation metrics
# print(f'Best KNN Model: {best_knn_model}')
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# # Save the model, scaler, and feature names
# joblib.dump(best_knn_model, 'heart_knn_model.pkl')
# joblib.dump(scaler, 'heart_scaler.pkl')
# heart_feature_names = X.columns
# print(X.columns)
# joblib.dump(heart_feature_names, 'heart_feature_names.pkl')



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

# # Save the model, scaler, and feature names
# joblib.dump(best_log_reg_model, 'heart_model.pkl')
# joblib.dump(scaler, 'heart_scaler.pkl')
# heart_feature_names = X.columns
# print(X.columns)
# joblib.dump(heart_feature_names, 'heart_feature_names.pkl')



from sklearn.svm import SVC
# Define the SVM classifier
svm = SVC()

# Set up the parameter grid for hyperparameter tuning
params_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV to find the best hyperparameters with stratified K-fold cross-validation
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
grid_svm = GridSearchCV(svm, params_svm, cv=skf, n_jobs=-1)
grid_svm.fit(X_train, y_train)
best_svm_model = grid_svm.best_estimator_

# Perform cross-validation on the best SVM model
cv_scores = cross_val_score(best_svm_model, X_train, y_train, cv=skf)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Predict and evaluate the model 
y_pred_svm = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
report = classification_report(y_test, y_pred_svm)
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Plot the confusion matrix
plt.figure(figsize=(10, 6))
plt.title('Confusion Matrix for Testing Data', fontsize=16)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm')
plt.xlabel('Predicted Value', fontsize=14)
plt.ylabel('Actual Value', fontsize=14)
plt.show()

# Print the evaluation metrics
print(f'Best SVM Model: {best_svm_model}')
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Calibrate the best SVM model using CalibratedClassifierCV
calibrated_svm = CalibratedClassifierCV(best_svm_model, method='sigmoid', cv=skf)
calibrated_svm.fit(X_train, y_train)
# Save the calibrated model
joblib.dump(calibrated_svm, 'calibrated_heart_svm_model.pkl')

# Save the model, scaler, and feature names
joblib.dump(best_svm_model, 'heart_model.pkl')
joblib.dump(scaler, 'heart_scaler.pkl')
heart_feature_names = X.columns
print(X.columns)
joblib.dump(heart_feature_names, 'heart_feature_names.pkl')
