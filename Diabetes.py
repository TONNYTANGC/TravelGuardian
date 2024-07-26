import joblib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import seaborn as sns
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import sklearn.feature_selection as fs

data = pd.read_csv('C:/Users/User/Desktop/TravelGuardian/\Dataset of Diabetes .csv')
# EDA
print(data.head())
print(data.info())

# Function to map CLASS to numerical value
def impact_CLASS(x):
    if x == 'N':
        return 0
    elif x == 'P':
        return 1
    else:
        return 2

# Apply the function 
data['CLASS'] = data['CLASS'].apply(impact_CLASS)

# Function to map gender to numerical value
def impact_gender(x):
    if x == 'F':
        return 0
    elif x == 'M':
        return 1

# Apply the function
data['Gender'] = data['Gender'].apply(impact_gender)

#Check Duplicated Values
print("\nDuplicated data:")
print(data.duplicated().sum())

missing_data=data.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage=(missing_data.sum()/data.shape[0])*100
print(f"The total percentage of missing data is {round(total_percentage,2)}%")
# Drop missing values
data.dropna(axis=0,inplace=True)

# Drop 'RecordID' column
data = data.drop(['ID','No_Pation'], axis=1)
print(data.head(2))

sns.countplot(x="CLASS",data=data)
plt.show()
cases=data.CLASS.value_counts()
print(f"There are {cases[0]} patients without risk of Diabetes, {cases[1]} patients with Predict Diabetes and {cases[2]} patients with risk of Diabetes")

fig=plt.figure(figsize=(15,20))
ax =fig.gca()
data.hist(ax = ax)
plt.show()

# Plot the correlation matrix
plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='inferno', annot=True)
plt.title('Correlation Matrix')
plt.show()

# Generate boxplot for the dataset
plt.subplots(figsize=(25, 15))
data.boxplot(patch_artist=True, sym="k.")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Boxplot of Features')
plt.show()

def detect_outlier(feature):
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return feature[(feature < lower_bound) | (feature > upper_bound)]

def cap_floor_outliers(data, feature):
    Q1 = np.percentile(data[feature], 25)
    Q3 = np.percentile(data[feature], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    data[feature] = np.where(data[feature] < lower_bound, lower_bound, data[feature])
    data[feature] = np.where(data[feature] > upper_bound, upper_bound, data[feature])

# Taking all the columns except the last one (assuming it's the target)
X = data.iloc[:, :-1]

# Apply cap and floor method to remove outliers
for feature in X.columns:
    cap_floor_outliers(data, feature)

# Verify that outliers are removed
for feature in X.columns:
    outliers = detect_outlier(data[feature])
    if not outliers.empty:
        print(f'Outliers still present in {feature}.')
    else:
        print(f'No outliers in {feature}.')

# Generate boxplot for the dataset after removing outliers
plt.subplots(figsize=(25, 15))
data.boxplot(patch_artist=True, sym="k.")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Boxplot of Features After Removing Outliers')
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
scaled_df = pd.DataFrame(data = scaled_data, columns = X.columns)
print(scaled_df.head())

label = data['CLASS']
encoder = LabelEncoder()
label = encoder.fit_transform(label)
X = scaled_df
y = label 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
print(X_train.shape, y_test.shape)
print(y_train.shape, y_test.shape)

xnew2=SelectKBest(f_classif, k=20).fit_transform(X, y)

df2 = fs.SelectKBest(k='all')
df2.fit(X, y)
names = X.columns.values[df2.get_support()]
scores = df2.scores_[df2.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns= ['Features','F_Scores'])
ns_df_sorted = ns_df.sort_values(['F_Scores','Features'], ascending = [False, True])
print(ns_df_sorted)

weights = np.linspace(0.05, 0.95, 20)

gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='accuracy',
    cv=15
)
grid_result = gsc.fit(X, y)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')

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
diabetes_feature_names = X.columns
print(X.columns)
joblib.dump(diabetes_feature_names, 'diabetes_feature_names.pkl')

# # Perform feature scaling on the training data
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
# hypertension_feature_names = X.columns
# # joblib.dump(hypertension_feature_names, 'diabetes_feature_names.pkl')