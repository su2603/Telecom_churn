# importing required libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.ioff()  # Turn off interactive mode
import seaborn as sns
import time
import math
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA

# Load the dataset
# Update this path to your dataset location
file_path = 'customer_churn_large_dataset.xlsx'
df = pd.read_excel(file_path)

# Exploratory Data Analysis (EDA)
print("First few rows of the dataset:")
print(df.head())

print("\nRandom sample of 8 rows:")
print(df.sample(8))

print(f"\nDataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns)

print("\nData info:")
df.info()

print("\nMissing values percentage:")
print(df.isna().sum()/len(df)*100)

print(f"\nNumber of duplicates: {df.duplicated().sum()}")

print("\nStatistical summary of numerical variables:")
print(df.describe())

print("\nStatistical summary of categorical variables:")
print(df.describe(include=['object']))

print("\nGender distribution:")
print(df['Gender'].value_counts())

print("\nLocation distribution:")
print(df['Location'].value_counts())

# Checking correlation between numerical variables
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Drop unnecessary columns
df.drop(columns=['CustomerID', 'Name'], axis=1, inplace=True)
print("\nColumns after dropping CustomerID and Name:")
print(df.columns)

# Check for typo's & suspicious values
print("\nChecking for typos and suspicious values:")
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in '{column}': {unique_values[:10] if len(unique_values) > 10 else unique_values}")
    print()

missing_values = df.isnull().sum()
data_types = df.dtypes

print('-'*50)
print("Missing values:")
print(missing_values)
print()

x = df.duplicated().sum()
print('-'*50)
print("Duplicate values:", x)
print()

y = df.shape
print('-'*50)
print("Shape of Dataset:", y)
print()

z = df.columns
print('-'*50)
print("Columns of Dataset:", z)
print()

print('-'*50)
print("\nData types:")
print(data_types)
print()

# Outliers Treatment
plt.figure(figsize=(10, 6))
df.select_dtypes(include=['float64', 'int64']).boxplot()
plt.title("Boxplot of Columns")
plt.xlabel("Columns")
plt.ylabel("Values")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('boxplot.png')
plt.close()

# Make a copy of cleaned data
df_cleaned = df.copy()

# Feature Encoding
df = pd.get_dummies(df, columns=['Gender', 'Location'], drop_first=True)
print("\nDataset after encoding:")
print(df.head())
print("\nData info after encoding:")
df.info()

# Checking Distribution of data
def sturges_rule(num_data_points):
    k = 1 + math.log2(num_data_points)
    return int(k)

# Example usage
num_data_points = len(df)
bins = sturges_rule(num_data_points)
print("\nNumber of bins according to Sturges' Rule:", bins)

# Check distribution of all continuous variables
continuous_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Exclude binary variables from the list
binary_vars = [var for var in continuous_vars if df[var].nunique() == 2]

# Exclude binary variables from the continuous variables
continuous_vars = [var for var in continuous_vars if var not in binary_vars]

# Plot the distribution of each continuous variable
for var in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=var, bins=bins, kde=True)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var}')
    plt.tight_layout()
    plt.savefig(f'distribution_{var}.png')
    plt.close()

# Check skewness of all continuous variables
print("\nSkewness of continuous variables:")
print(df[continuous_vars].skew())

# Updated correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('correlation_matrix_updated.png')
plt.close()

# Diving data into train and test set
x = df.drop("Churn", axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("\nTraining/Test split shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Feature Scaling
columns_to_check = df.columns
columns_needs_to_be_scaled = []

for column in columns_to_check:
    if column in df.columns:  # Check if column exists in df
        if (df[column] > 1).any() or (df[column] < 0).any():
            columns_needs_to_be_scaled.append(column)

print("\nColumns with values greater than 1 or less than 0:")
print(columns_needs_to_be_scaled)

columns_to_scale = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

scaler = MinMaxScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

print("\nX_train after scaling:")
print(X_train.head())
print("\nX_test after scaling:")
print(X_test.head())

# Check for class imbalance
class_counts = df['Churn'].value_counts()
print("\nClass distribution:")
print(class_counts)

plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# Feature Selection using Random Forest Feature Importance Method
rf_classifier = RandomForestClassifier(n_jobs=-1, random_state=42)
rf_classifier.fit(X_train, y_train)
importances = rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\nFeature importance:")
print(feature_importance_df)

# Check optimal number of features
sorted_indices = np.argsort(importances)[::-1]
cumulative_importance = np.cumsum(importances[sorted_indices])

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(importances) + 1), cumulative_importance, 'b-')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Importance of Features')
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_importance.png')
plt.close()

# Checking multicollinearity of X_train
X_train_for_vif = X_train.copy()

# Make sure all values are numeric and finite
X_train_for_vif = X_train_for_vif.select_dtypes(include=['float64', 'int64'])
X_train_for_vif = X_train_for_vif.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

# Only calculate VIF if we have enough features left
if X_train_for_vif.shape[1] > 1:
    vif = pd.DataFrame()
    vif["Variable"] = X_train_for_vif.columns
    vif["VIF"] = [variance_inflation_factor(X_train_for_vif.values, i) for i in range(X_train_for_vif.shape[1])]
    vif = vif.sort_values(by='VIF', ascending=False)
    
    print("\nVariance Inflation Factors (VIF):")
    print(vif)
else:
    print("\nNot enough numeric features to calculate VIF after filtering.")


# Model Building: Machine Learning Algorithms
algorithms = [
    LogisticRegression(n_jobs=-1, random_state=42, max_iter=1000),
    DecisionTreeClassifier(random_state=42),
    KNeighborsClassifier(n_jobs=-1),
    GaussianNB(),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    RandomForestClassifier(n_jobs=-1, random_state=42),
    XGBClassifier(n_jobs=-1, random_state=42),
    SVC(random_state=42, probability=True)
]

# Initialize the results dictionary for training data
results_train = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'Building Time (s)': []
}

# Apply the algorithms and calculate performance metrics for training data
print("\nTraining and evaluating ML models...")
for algorithm in algorithms:
    start_time = time.time()  # Start timer

    algorithm_name = type(algorithm).__name__
    algorithm.fit(X_train, y_train)
    y_train_pred = algorithm.predict(X_train)

    accuracy = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_train, y_train_pred, average='weighted')
    recall = recall_score(y_train, y_train_pred, average='weighted')
    f1 = f1_score(y_train, y_train_pred, average='weighted')

    end_time = time.time()  # End timer
    building_time = end_time - start_time

    results_train['Algorithm'].append(algorithm_name)
    results_train['Accuracy'].append(accuracy)
    results_train['Precision'].append(precision)
    results_train['Recall'].append(recall)
    results_train['F1-score'].append(f1)
    results_train['Building Time (s)'].append(building_time)

# Create a dataframe for the training data results
df_train = pd.DataFrame(results_train)
print("\nTraining metrics:")
print(df_train)

results_test = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': []
}

# Apply the algorithms and calculate performance metrics for test data
for algorithm in algorithms:
    algorithm_name = type(algorithm).__name__
    y_test_pred = algorithm.predict(X_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    results_test['Algorithm'].append(algorithm_name)
    results_test['Accuracy'].append(accuracy)
    results_test['Precision'].append(precision)
    results_test['Recall'].append(recall)
    results_test['F1-score'].append(f1)

# Create a dataframe for the test data results
df_test = pd.DataFrame(results_test)
print("\nTest metrics:")
print(df_test)

# Using selected important features
print("\nSelecting most important features...")
X_train_select = X_train[['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']]
X_test_select = X_test[['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']]

print('X_train columns:', X_train_select.columns)
print('-'*120)
print('X_test columns:', X_test_select.columns)

algorithms = [
    LogisticRegression(n_jobs=-1, random_state=42, max_iter=1000),
    DecisionTreeClassifier(random_state=42),
    KNeighborsClassifier(n_jobs=-1),
    GaussianNB(),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    RandomForestClassifier(n_jobs=-1, random_state=42),
    XGBClassifier(n_jobs=-1, random_state=42),
    SVC(random_state=42, probability=True)
]

# Initialize the results dictionary for training data with selected features
results_train = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'Building Time (s)': []
}

# Apply the algorithms and calculate performance metrics for training data with selected features
print("\nTraining and evaluating ML models with selected features...")
for algorithm in algorithms:
    start_time = time.time()  # Start timer

    algorithm_name = type(algorithm).__name__
    algorithm.fit(X_train_select, y_train)
    y_train_pred = algorithm.predict(X_train_select)

    accuracy = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_train, y_train_pred, average='weighted')
    recall = recall_score(y_train, y_train_pred, average='weighted')
    f1 = f1_score(y_train, y_train_pred, average='weighted')

    end_time = time.time()  # End timer
    building_time = end_time - start_time

    results_train['Algorithm'].append(algorithm_name)
    results_train['Accuracy'].append(accuracy)
    results_train['Precision'].append(precision)
    results_train['Recall'].append(recall)
    results_train['F1-score'].append(f1)
    results_train['Building Time (s)'].append(building_time)

# Create a dataframe for the training data results with selected features
df_train = pd.DataFrame(results_train)
print("\nTraining metrics with selected features:")
print(df_train)

results_test = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': []
}

# Apply the algorithms and calculate performance metrics for test data with selected features
for algorithm in algorithms:
    algorithm_name = type(algorithm).__name__
    y_test_pred = algorithm.predict(X_test_select)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    results_test['Algorithm'].append(algorithm_name)
    results_test['Accuracy'].append(accuracy)
    results_test['Precision'].append(precision)
    results_test['Recall'].append(recall)
    results_test['F1-score'].append(f1)

# Create a dataframe for the test data results with selected features
df_test = pd.DataFrame(results_test)
print("\nTest metrics with selected features:")
print(df_test)

# Model Building: Neural Network
print("\nBuilding Neural Network models...")

# Define the EarlyStopping and ModelCheckpoint callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)
checkpoint = ModelCheckpoint('ChurnClassifier.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Architecture I
print("\nTraining Neural Network Architecture I...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train_select.shape[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_select, y_train, 
    epochs=20,  # Reduced for faster execution
    batch_size=128, 
    validation_split=0.3, 
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Architecture II
print("\nTraining Neural Network Architecture II...")
model = Sequential()

# layers
model.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=X_train_select.shape[1]))
model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_select, y_train, 
    epochs=20,  # Reduced for faster execution
    batch_size=128, 
    validation_split=0.3, 
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Architecture III
print("\nTraining Neural Network Architecture III...")
model = Sequential()

# layers
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=X_train_select.shape[1]))
model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_select, y_train, 
    epochs=20,  # Reduced for faster execution
    batch_size=128, 
    validation_split=0.3, 
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Architecture IV
print("\nTraining Neural Network Architecture IV...")
model = Sequential()

# Input layer with BatchNormalization and Activation (ReLU)
model.add(Dense(10, input_dim=X_train_select.shape[1], kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# First hidden layer with BatchNormalization, Activation (ReLU), and Dropout
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))  # 20% dropout

# Second hidden layer with BatchNormalization, Activation (ReLU), and Dropout
model.add(Dense(5, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))  # 10% dropout

# Output layer with Sigmoid activation
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_select, y_train, 
    epochs=20,  # Reduced for faster execution 
    batch_size=128, 
    validation_split=0.3, 
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Architecture V
print("\nTraining Neural Network Architecture V...")
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_select.shape[1]),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_select, y_train, 
    epochs=20,  # Reduced for faster execution
    batch_size=128, 
    validation_split=0.2, 
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Ensembles of Random Forest
print("\nBuilding ensemble models...")
# Initialize base estimator (Random Forest)
base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize models
adaboost_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
gradient_boost_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
xgboost_model = XGBClassifier(n_estimators=50, random_state=42)

# Initialize lists to store metrics
models = ['AdaBoost', 'Gradient Boosting', 'XGBoost']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Building Time']
results_train = {metric: [] for metric in metrics}
results_test = {metric: [] for metric in metrics}

# Train and evaluate models
for model in [adaboost_model, gradient_boost_model, xgboost_model]:
    start_time = time.time()
    model.fit(X_train_select, y_train)
    end_time = time.time()

    # Predict on the training set
    y_train_pred = model.predict(X_train_select)

    # Calculate metrics on training data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='weighted')
    recall_train = recall_score(y_train, y_train_pred, average='weighted')
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    building_time = end_time - start_time

    # Predict on the test set
    y_test_pred = model.predict(X_test_select)

    # Calculate metrics on test data
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')

    # Append metrics to the results dictionaries
    results_train['Accuracy'].append(accuracy_train)
    results_train['Precision'].append(precision_train)
    results_train['Recall'].append(recall_train)
    results_train['F1 Score'].append(f1_train)
    results_train['Building Time'].append(building_time)

    results_test['Accuracy'].append(accuracy_test)
    results_test['Precision'].append(precision_test)
    results_test['Recall'].append(recall_test)
    results_test['F1 Score'].append(f1_test)
    results_test['Building Time'].append(building_time)

# Create DataFrames from the results
results_train_df = pd.DataFrame(results_train, index=models)
results_test_df = pd.DataFrame(results_test, index=models)

# Display the DataFrames
print("\nEnsemble Models - Training Data Results:")
print(results_train_df)
print("\nEnsemble Models - Test Data Results:")
print(results_test_df)

# Model Building: PCA
print("\nPerforming PCA analysis...")
df_cleaned = pd.get_dummies(df_cleaned, columns=['Gender', 'Location'], drop_first=True)
x = df_cleaned.drop('Churn', axis=1)
y = df_cleaned['Churn']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled numpy arrays back to DataFrames
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nHead of scaled training data:")
print(X_train_df.head())
print("\nHead of scaled test data:")
print(X_test_df.head())

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
print(f"\nShape after PCA transformation: {X_train_pca.shape}")

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot the scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid()
plt.tight_layout()
plt.savefig('scree_plot.png')
plt.close()

# Select optimal number of components
n_components = 8  # Based on scree plot analysis
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nShape after PCA with {n_components} components:")
print(f"X_train_pca: {X_train_pca.shape}, X_test_pca: {X_test_pca.shape}")

# Apply ML algorithms on PCA-transformed data
algorithms = [
    LogisticRegression(n_jobs=-1, random_state=42, max_iter=1000),
    DecisionTreeClassifier(random_state=42),
    KNeighborsClassifier(n_jobs=-1),
    GaussianNB(),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    RandomForestClassifier(n_jobs=-1, random_state=42),
    XGBClassifier(n_jobs=-1, random_state=42),
    SVC(random_state=42, probability=True)
]

# Initialize the results dictionary for training data with PCA
results_train = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'Building Time (s)': []
}

# Apply the algorithms and calculate performance metrics for training data using X_train_pca
print("\nTraining and evaluating ML models with PCA...")
for algorithm in algorithms:
    start_time = time.time()  # Start timer

    algorithm_name = type(algorithm).__name__
    algorithm.fit(X_train_pca, y_train)
    y_train_pred = algorithm.predict(X_train_pca)

    accuracy = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_train, y_train_pred, average='weighted')
    recall = recall_score(y_train, y_train_pred, average='weighted')
    f1 = f1_score(y_train, y_train_pred, average='weighted')

    end_time = time.time()  # End timer
    building_time = end_time - start_time

    results_train['Algorithm'].append(algorithm_name)
    results_train['Accuracy'].append(accuracy)
    results_train['Precision'].append(precision)
    results_train['Recall'].append(recall)
    results_train['F1-score'].append(f1)
    results_train['Building Time (s)'].append(building_time)

# Create a DataFrame for the training data results with PCA
df_train = pd.DataFrame(results_train)
print("\nTraining metrics with PCA:")
print(df_train)

results_test = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': []
}

# Apply the algorithms and calculate performance metrics for test data using X_test_pca
for algorithm in algorithms:
    algorithm_name = type(algorithm).__name__
    y_test_pred = algorithm.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    results_test['Algorithm'].append(algorithm_name)
    results_test['Accuracy'].append(accuracy)
    results_test['Precision'].append(precision)
    results_test['Recall'].append(recall)
    results_test['F1-score'].append(f1)

# Create a DataFrame for the test data results with PCA
df_test = pd.DataFrame(results_test)
print("\nTest metrics with PCA:")
print("\nTest metrics with PCA:")
print(df_test)

# Select the best performing algorithm (XGBoost based on previous results)
best_model = XGBClassifier(n_jobs=-1, random_state=42)
best_model.fit(X_train_select, y_train)

# ROC Curve for the best model
y_test_prob = best_model.predict_proba(X_test_select)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
auc = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# Confusion Matrix for the best model
y_test_pred = best_model.predict(X_test_select)
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Hyperparameter Tuning using GridSearchCV for XGBoost
print("\nPerforming hyperparameter tuning for XGBoost...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_select, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

# Train the final model with optimized hyperparameters
final_model = XGBClassifier(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_select, y_train)

# Save the final model to disk
print("\nSaving the final model to disk...")
joblib.dump(final_model, 'final_xgboost_model.pkl')
joblib.dump(scaler, 'scaler_model.pkl')

# Evaluate the final model on the test set
y_test_pred = final_model.predict(X_test_select)
print("\nFinal model performance on test set:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

# Feature importance for the final model
feature_importances = final_model.feature_importances_
feature_names = X_train_select.columns

# Create a DataFrame for feature importance and sort by importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature importance for the final model:")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for XGBoost Model')
plt.tight_layout()
plt.savefig('final_feature_importance.png')
plt.close()

# Cross-validation for the final model
print("\nPerforming cross-validation for the final model...")
cv_scores = cross_val_score(final_model, X_train_select, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Function to predict churn for new customers
def predict_customer_churn(customer_data, model, scaler):
    """
    Predict whether a customer will churn or not.
    
    Parameters:
    -----------
    customer_data : pandas DataFrame
        DataFrame containing customer information
    model : trained machine learning model
        The final model used for prediction
    scaler : fitted scaler
        The scaler used for feature scaling
    
    Returns:
    --------
    prediction : int
        0 if customer is predicted not to churn, 1 if predicted to churn
    probability : float
        The probability of the customer churning
    """
    # Select relevant features
    features = ['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']
    customer_features = customer_data[features]
    
    # Scale the features
    scaled_features = scaler.transform(customer_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[:, 1]
    
    return prediction[0], probability[0]

# Example of using the prediction function
print("\nExample of predicting churn for a new customer:")
new_customer = pd.DataFrame({
    'Age': [35],
    'Monthly_Bill': [70.5],
    'Total_Usage_GB': [50],
    'Subscription_Length_Months': [12]
})

# Load the saved model and scaler
loaded_model = joblib.load('final_xgboost_model.pkl')
loaded_scaler = joblib.load('scaler_model.pkl')

# Make prediction
prediction, probability = predict_customer_churn(new_customer, loaded_model, loaded_scaler)
print(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
print(f"Probability of churning: {probability:.4f}")

# Model interpretation using SHAP values (optional, requires shap library)
try:
    import shap
    print("\nGenerating SHAP values for model interpretation...")
    # Create a small sample of the test data for SHAP analysis
    X_test_sample = X_test_select.sample(min(100, len(X_test_select)), random_state=42)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=X_test_sample.columns, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    plt.close()
    
    # Dependence plots for top features
    for feature in feature_importance_df['Feature'].head(2):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test_sample, feature_names=X_test_sample.columns, show=False)
        plt.title(f'SHAP Dependence Plot for {feature}')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature}.png')
        plt.close()
except ImportError:
    print("\nSHAP library not installed. Skipping model interpretation with SHAP values.")

# Create a simple customer segmentation based on churn probability
print("\nCreating customer segmentation based on churn probability...")

# Predict probabilities for all test customers
test_probs = final_model.predict_proba(X_test_select)[:, 1]

# Create a DataFrame with customer data and churn probabilities
customer_segments = X_test.copy()
customer_segments['Churn_Probability'] = test_probs
customer_segments['Actual_Churn'] = y_test.values

# Define segments based on churn probability
def assign_segment(prob):
    if prob < 0.3:
        return 'Low Risk'
    elif prob < 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'

customer_segments['Risk_Segment'] = customer_segments['Churn_Probability'].apply(assign_segment)

# Display segment distribution
segment_counts = customer_segments['Risk_Segment'].value_counts()
print("\nCustomer segment distribution:")
print(segment_counts)

# Plot segment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Risk_Segment', data=customer_segments, palette='viridis')
plt.title('Customer Distribution by Churn Risk Segment')
plt.ylabel('Count')
plt.xlabel('Risk Segment')
plt.tight_layout()
plt.savefig('customer_segments.png')
plt.close()

# Calculate actual churn rate by segment
segment_churn_rates = customer_segments.groupby('Risk_Segment')['Actual_Churn'].mean().sort_values(ascending=False)
print("\nActual churn rate by segment:")
print(segment_churn_rates)

# Plot actual churn rate by segment
plt.figure(figsize=(10, 6))
segment_churn_rates.plot(kind='bar', color='darkred')
plt.title('Actual Churn Rate by Risk Segment')
plt.ylabel('Churn Rate')
plt.xlabel('Risk Segment')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('segment_churn_rates.png')
plt.close()

# Summary statistics by segment
segment_stats = customer_segments.groupby('Risk_Segment').agg({
    'Monthly_Bill': 'mean',
    'Total_Usage_GB': 'mean',
    'Age': 'mean',
    'Subscription_Length_Months': 'mean',
    'Churn_Probability': 'mean',
    'Actual_Churn': 'mean'
}).round(2)

print("\nSummary statistics by segment:")
print(segment_stats)

# Potential revenue at risk calculation
customer_segments['Monthly_Revenue'] = X_test['Monthly_Bill']  # Use original values
revenue_at_risk = customer_segments.groupby('Risk_Segment').agg({
    'Monthly_Revenue': 'sum',
    'Churn_Probability': 'mean'
})
revenue_at_risk['Expected_Revenue_Loss'] = revenue_at_risk['Monthly_Revenue'] * revenue_at_risk['Churn_Probability']
revenue_at_risk = revenue_at_risk.sort_values('Expected_Revenue_Loss', ascending=False)

print("\nPotential monthly revenue at risk by segment:")
print(revenue_at_risk)

# Plot potential revenue at risk
plt.figure(figsize=(10, 6))
sns.barplot(x=revenue_at_risk.index, y='Expected_Revenue_Loss', data=revenue_at_risk)
plt.title('Expected Monthly Revenue Loss by Segment')
plt.ylabel('Expected Monthly Revenue Loss ($)')
plt.xlabel('Risk Segment')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('revenue_at_risk.png')
plt.close()

# Generate business recommendations based on analysis
print("\n" + "="*50)
print("BUSINESS RECOMMENDATIONS")
print("="*50)

print("\n1. Focus retention efforts on the High Risk segment which represents approximately",
      f"{segment_counts.get('High Risk', 0) / len(customer_segments) * 100:.1f}% of customers",
      f"but accounts for ${revenue_at_risk.loc['High Risk', 'Expected_Revenue_Loss']:.2f} in monthly expected revenue loss.")

print("\n2. Analyze the key factors driving churn in the High Risk segment:")
for feature in feature_importance_df['Feature'].head(3):
    high_risk_avg = customer_segments[customer_segments['Risk_Segment'] == 'High Risk'][feature].mean()
    low_risk_avg = customer_segments[customer_segments['Risk_Segment'] == 'Low Risk'][feature].mean()
    print(f"   - {feature}: High Risk avg = {high_risk_avg:.2f}, Low Risk avg = {low_risk_avg:.2f}")

print("\n3. Implement targeted retention strategies:")
print("   - Offer discounts or loyalty rewards to high-risk customers")
print("   - Provide personalized usage recommendations based on customer patterns")
print("   - Implement proactive customer service outreach for at-risk customers")

print("\n4. Monitor the Medium Risk segment to prevent migration to High Risk")

print("\n5. Regularly retrain the model to adapt to changing customer behaviors and market conditions")

# Final metrics summary
print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"AUC-ROC: {auc:.4f}")

print("\nModel training complete! The final XGBoost model has been saved as 'final_xgboost_model.pkl'")
print("The feature scaling model has been saved as 'scaler_model.pkl'")
print("\nUse the predict_customer_churn() function to make predictions for new customers.")

# End of script
print("\nCustomer churn analysis complete!")