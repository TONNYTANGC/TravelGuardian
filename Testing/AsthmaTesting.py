# Importing libraries
from collections import Counter
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Loading data
data = pd.read_csv("C:/Users/User/Desktop/TravelGuardian/asthma_dataset.csv")

# EDA
print(data.head())
print(data.info())

# Check duplicated values
print("\nDuplicated data:")
print(data.duplicated().sum())

missing_data = data.isnull().sum()
print("\nMissing data:")
print(missing_data)
total_percentage = (missing_data.sum() / data.shape[0]) * 100
print(f"The total percentage of missing data is {round(total_percentage, 2)}%")

data = data.drop(['Patient_ID', 'Medication'], axis=1)

data['Asthma_Diagnosis'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')

# Adding titles and labels
plt.title('Value Counts of Target Variable', fontsize=16)
plt.xlabel('Target', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Display the plot
plt.show()

numerical_features = data.select_dtypes(include=['int', 'float']).columns.to_list()
categorical_features = data.select_dtypes(include=['object', 'category']).columns.to_list()

sns.set_style('darkgrid')
colors = sns.color_palette(palette='bright', n_colors=len(numerical_features))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3.2))
ax = ax.flat

for i, feature in enumerate(numerical_features):
    sns.kdeplot(data, x=feature, fill=True, color=colors[i], ax=ax[i])
    sns.histplot(data, x=feature, stat='density', fill=False, color=colors[i], ax=ax[i])
    ax[i].set_xlabel('')
    ax[i].set_title(feature, fontsize=11, fontweight='bold', color='black')
 
fig.suptitle("Distribution of numerical variables", fontsize=12, fontweight='bold', color='firebrick')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3.2))
ax = ax.flat

for i, feature in enumerate(numerical_features):
    sm.qqplot(data[feature], line='q', ax=ax[i], lw=2.1)
    ax[i].set_title(feature, fontsize=11, fontweight='bold', color='black')
 
fig.suptitle("Q-Q Plots", fontsize=13, fontweight='bold', color='firebrick')
fig.tight_layout()
fig.show()

# Function to map gender to numerical value
def impact_gender(x):
    if x == 'Female':
        return 0
    elif x == 'Male':
        return 1

# Apply the function
data['Gender'] = data['Gender'].apply(impact_gender)

# Function to map smoker status to numerical value
def impact_smoking(x):
    if x == 'Non-Smoker':
        return 0
    elif x == 'Ex-Smoker':
        return 1
    elif x == 'Current Smoker':
        return 2
    
# Apply the function
data['Smoking_Status'] = data['Smoking_Status'].apply(impact_smoking)

print(data.head())

# Separate features and target variable
TARGET = 'Asthma_Diagnosis'
TEST_SIZE = 0.2
SEED = 123

X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

print(f"y_train: {Counter(y_train)}")
print(f"y_test: {Counter(y_test)}")

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(f"y_train: {Counter(y_train)}")
print(f"y_test: {Counter(y_test)}")

numerical_predictors = X_train.select_dtypes(include=['float', 'int']).columns.to_list()
categorical_predictors = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

# Transformations for the following models: LogisticRegression, SVM and KNN.
transformer = [('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_predictors), ('scaler', MinMaxScaler(), numerical_predictors)]

# Preprocessor for the following models: LogisticRegression, SVM and KNN.
preprocessor = ColumnTransformer(transformers=transformer, remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False).set_output(transform='pandas')

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

clf1 = CalibratedClassifierCV(LogisticRegression(random_state=SEED))
clf2 = CalibratedClassifierCV(KNeighborsClassifier(n_jobs=-1))
clf3 = CalibratedClassifierCV(SVC(random_state=SEED, probability=True))

MODELS1 = [clf1, clf2, clf3]
best_model = None
best_score = 0

# Training !!!
for model in tqdm(MODELS1):
    name = type(model).__name__

    model.fit(X_train_prep.to_numpy(dtype=np.float32), y_train)

    y_pred_train = model.predict(X_train_prep.to_numpy(dtype=np.float32))
    y_pred_test = model.predict(X_test_prep.to_numpy(dtype=np.float32))
        
    score_train = fbeta_score(y_train, y_pred_train, beta=2)
    score_test = fbeta_score(y_test, y_pred_test, beta=2)
    
    print("==" * 30)
    print(f"\033[1;33m {name} \033[0;m :\n") 
    print(f' F2 Train: {score_train:.4f} |', 
          f'F2 Test: {score_test:.4f}\n')
    print("==" * 30)
    
    # Save the model if it's the best one found so far
    if score_test > best_score:
        best_score = score_test
        best_model = model
        joblib.dump(best_model, f'asthma_best_model.pkl')

# Save the scaler and feature names
joblib.dump(preprocessor.named_transformers_['scaler'], 'asthma_scaler.pkl')
joblib.dump(X.columns, 'asthma_feature_names.pkl')
print(best_model)