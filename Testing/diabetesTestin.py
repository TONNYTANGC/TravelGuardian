import joblib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('C:/Users/User/Desktop/TravelGuardian/diabetes.csv')
# EDA
print(data.head())
print(data.info())

#Check Duplicated Values
print("\nDuplicated data:")
print(data.duplicated().sum())
data.drop_duplicates(inplace=True)

missing_data=data.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage=(missing_data.sum()/data.shape[0])*100
print(f"The total percentage of missing data is {round(total_percentage,2)}%")

sns.countplot(x="Outcome",data=data)
plt.show()
cases=data.Outcome.value_counts()
print(f"There are {cases[0]} patients without risk of Diabetes and {cases[1]} patients with risk of Diabetes")

fig=plt.figure(figsize=(15,20))
ax =fig.gca()
data.hist(ax = ax)
plt.show()

sns.set_theme(context='poster')
plt.figure(figsize=(10,7))
plt.title('Glucose distribution based on Diabetes Risk', color="Black",fontsize=25)

sns.distplot(data[data['Outcome'] == 0]['Glucose'], label='No risk of Diabetes')
sns.distplot(data[data['Outcome'] == 1]['Glucose'], label = 'Risk of Diabetes')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

num= data[data["SkinThickness"]==0]
num1= data[data["BloodPressure"]==0]
num2= data[data["Glucose"]==0]
num3= data[data["Insulin"]==0]
num4= data[data["BMI"]==0]
print(num.shape,num1.shape,num2.shape,num3.shape,num4.shape)

# Replace 0 with mean values 
data[['Glucose', 'BloodPressure', 'BMI','Insulin','SkinThickness']] = data[['Glucose', 'BloodPressure', 'BMI','Insulin','SkinThickness']].replace(0, np.nan)
data[['Glucose', 'BloodPressure', 'BMI','Insulin','SkinThickness']] = data[['Glucose', 'BloodPressure', 'BMI','Insulin','SkinThickness']].fillna(data[['Glucose', 'BloodPressure', 'BMI','Insulin','SkinThickness']].mean())

# Split the data into training and testing sets
X = data.drop(columns=['Outcome','BloodPressure','SkinThickness'])
y = data['Outcome']

# balance the class distribution
ros = RandomOverSampler(random_state=41)
X_ros,y_ros = ros.fit_resample(X,y)

X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# # Perform feature scaling on the training data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = Sequential([

#     Dense(150,activation='relu',name='a1',input_shape=(X_train.shape[1],)),
#     Dense(100,activation='relu',name='a2'),
#     Dense(50,activation='relu',name='a3'),
#     Dense(25,activation='relu',name='a4'),
#     Dense(1,activation='sigmoid',name='a5')
    
# ]
# )
# model.compile(
#     loss = tf.keras.losses.BinaryCrossentropy(),
#     optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
#     metrics=['accuracy']

# )
# # Train the model
# model.fit(X_train, y_train, epochs=75, batch_size=32)

# # Make predictions on the test set
# y_pred_prob = model.predict(X_test)
# y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Print the evaluation metrics
# print(f'Neural Network Model: {model}')
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

# # Save the model and scaler
# model.save('diabetes_nn_model.h5')
# joblib.dump(scaler, 'diabetes_scaler.pkl')

# # Save the feature names
# hypertension_feature_names = X_ros.columns
# joblib.dump(hypertension_feature_names, 'diabetes_feature_names.pkl')


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(knn, params_knn, cv=6, n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn_model = grid_knn.best_estimator_

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
joblib.dump(best_knn_model, 'diabetes_knn_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')
heart_feature_names = X.columns
print(X.columns)
joblib.dump(heart_feature_names, 'diabetes_feature_names.pkl')


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