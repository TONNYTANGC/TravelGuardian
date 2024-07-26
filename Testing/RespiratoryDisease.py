import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Model Accuracies
ml_accuracies = dict()

df = pd.read_csv('C:/Users/User/Desktop/TravelGuardian/cancer patient data sets.csv')

# Index Column now refers to patient
df.drop("Patient Id", axis=1, inplace=True)

# Cleaning column names
df.rename(columns=str.lower, inplace=True)
df.rename(columns={col: col.replace(" ", "_") for col in df.columns}, inplace=True)

#EDA
print(df)
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum())

print('Cancer Levels: ', df['level'].unique())

# Replacing levels of numeric int
mapping = {'High': 100, 'Medium': 50, 'Low': 0}
df["level"].replace(mapping, inplace=True)
print('Cancer Levels: ', df['level'].unique()) 

# Showing data
X = df.drop(columns='level')
y = df.level

print(X.head())
print(y[:5])

plt.figure(figsize=(6, 6))
plt.title('Training Data', fontsize=20)
plt.pie(df.level.value_counts(),
    labels=mapping.keys(),
    colors=['#FAC500','#0BFA00', '#0066FA','#FA0000'], 
    autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum(df.level.value_counts() /100)),
    explode=tuple(0.01 for i in range(3)),
    textprops={'fontsize': 20}
)
plt.show()

# plt.figure(figsize=(20,15))
# sns.heatmap(df.corr(), annot=True, cmap=plt.cm.PuBu)
# plt.show()

plt.figure(figsize = (20, 40))

for i in range(24):
    plt.subplot(16, 2, i+1)
    sns.distplot(df.iloc[:, i], color = 'red')
    plt.grid()
plt.show()



# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=24)
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
X = df.drop(['index', 'level','occupational_hazards', 'genetic_risk','age','gender','swallowing_difficulty','snoring','dry_cough','weight_loss','clubbing_of_finger_nails'],axis=1)
y = df.level
# Display the updated dataframe
print(X.info())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40)
print(f'Shapes - X Training: {X_train.shape} and X Testing {X_test.shape}')
print(f'Shapes - Y Training: {y_train.shape} and Y Testing {y_test.shape}')

print(f'\nTraining output counts\n{y_train.value_counts()}')

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
# Define the parameter grid
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # Smoothing parameter
    'binarize': [0.0, 0.5, 1.0, 1.5]            # Threshold for binarizing (preprocessing step)
}

# Initialize the BernoulliNB model
nb = BernoulliNB()

# Set up GridSearchCV
grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_nb = grid_search.best_estimator_

# Print the best hyperparameters
print('Best Hyperparameters:', grid_search.best_params_)

# Predict with the best model
nb_pred = best_nb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, nb_pred)
print(f'Accuracy: {accuracy:.4f}')

# Generate confusion matrix
cm = confusion_matrix(y_test, nb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - BernoulliNB')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
print('Classification Report:')
print(classification_report(y_test, nb_pred))

# Save the best model and scaler
joblib.dump(best_nb, 'respiratory_model.pkl')
joblib.dump(scaler, 'respiratory_scaler.pkl')
respiratory_feature_names = X.columns.tolist()
joblib.dump(respiratory_feature_names, 'respiratory_feature_names.pkl')
joblib.dump(respiratory_feature_names, 'respiratory_feature_names.pkl')