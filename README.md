# Customer Churn Analysis s

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
   - [Data Loading](#data-loading)
   - [Data Cleaning](#data-cleaning)
   - [Feature Engineering](#feature-engineering)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Statistical Summary](#statistical-summary)
   - [Correlation Analysis](#correlation-analysis)
   - [Distribution Analysis](#distribution-analysis)
4. [Feature Selection](#feature-selection)
   - [Random Forest Feature Importance](#random-forest-feature-importance)
   - [Multicollinearity Check](#multicollinearity-check)
5. [Model Building](#model-building)
   - [Traditional Machine Learning Models](#traditional-machine-learning-models)
   - [Feature-Reduced Models](#feature-reduced-models)
   - [Neural Network Models](#neural-network-models)
   - [Ensemble Models](#ensemble-models)
   - [PCA Models](#pca-models)
6. [Model Evaluation](#model-evaluation)
   - [Metrics Comparison](#metrics-comparison)
   - [ROC Curve Analysis](#roc-curve-analysis)
   - [Confusion Matrix](#confusion-matrix)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Final Model](#final-model)
   - [Model Performance](#model-performance)
   - [Feature Importance](#feature-importance)
   - [Cross-Validation](#cross-validation)
9. [Model Deployment](#model-deployment)
   - [Prediction Function](#prediction-function)
   - [Model Persistence](#model-persistence)
10. [Customer Segmentation](#customer-segmentation)
    - [Risk Segments](#risk-segments)
    - [Segment Analysis](#segment-analysis)
    - [Revenue Impact](#revenue-impact)
11. [Business Recommendations](#business-recommendations)
12. [Future Work](#future-work)

## Introduction

This documentation provides an in-depth analysis of customer churn prediction using machine learning algorithms. Customer churn refers to the phenomenon where customers stop using a service or product. Predicting which customers are likely to churn allows businesses to take proactive measures to retain them.

The analysis follows a comprehensive machine learning workflow:

1. Data preparation and cleaning
2. Exploratory data analysis
3. Feature selection
4. Model building and comparison
5. Hyperparameter tuning
6. Model evaluation and interpretation
7. Customer segmentation based on churn risk
8. Business recommendations

The code is designed to provide both analytical insights and practical business applications.

## Data Preparation

### Data Loading

The dataset is loaded from an Excel file named 'customer_churn_large_dataset.xlsx'. It contains customer information including demographic data, subscription details, usage patterns, and churn status.

```python
# Load the dataset
file_path = 'customer_churn_large_dataset.xlsx'
df = pd.read_excel(file_path)
```

### Data Cleaning

The cleaning process includes:

1. **Handling Missing Values**: The code checks for missing values in the dataset and reports the percentage.
   ```python
   print("\nMissing values percentage:")
   print(df.isna().sum()/len(df)*100)
   ```

2. **Removing Duplicates**: Duplicate records are identified and counted.
   ```python
   print(f"\nNumber of duplicates: {df.duplicated().sum()}")
   ```

3. **Dropping Unnecessary Columns**: Non-predictive columns like CustomerID and Name are removed.
   ```python
   df.drop(columns=['CustomerID', 'Name'], axis=1, inplace=True)
   ```

4. **Checking for Suspicious Values**: Each column is examined for unusual or suspicious values.
   ```python
   for column in df.columns:
       unique_values = df[column].unique()
       print(f"Unique values in '{column}': {unique_values[:10] if len(unique_values) > 10 else unique_values}")
   ```

### Feature Engineering

1. **Categorical Variable Encoding**: One-hot encoding is applied to categorical variables.
   ```python
   df = pd.get_dummies(df, columns=['Gender', 'Location'], drop_first=True)
   ```

2. **Feature Scaling**: Numerical features are scaled using MinMaxScaler to ensure all values are between 0 and 1.
   ```python
   columns_to_scale = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
   scaler = MinMaxScaler()
   X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
   X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
   ```

## Exploratory Data Analysis

### Statistical Summary

The code provides comprehensive statistical summaries for both numerical and categorical variables:

```python
print("\nStatistical summary of numerical variables:")
print(df.describe())

print("\nStatistical summary of categorical variables:")
print(df.describe(include=['object']))
```

Key statistics examined include:
- Mean, median, standard deviation for numerical variables
- Frequency counts for categorical variables
- Distribution of the target variable (Churn)

### Correlation Analysis

A correlation matrix is generated to identify relationships between variables:

```python
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
```

The correlation analysis helps identify potential multicollinearity issues and strong predictors of churn.

### Distribution Analysis

The code examines the distribution of continuous variables using histograms:

```python
for var in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=var, bins=bins, kde=True)
```

Skewness is also calculated to assess the symmetry of distributions:

```python
print("\nSkewness of continuous variables:")
print(df[continuous_vars].skew())
```

Additionally, outliers are identified using box plots:

```python
plt.figure(figsize=(10, 6))
df.select_dtypes(include=['float64', 'int64']).boxplot()
```

## Feature Selection

### Random Forest Feature Importance

A Random Forest classifier is used to identify the most important features:

```python
rf_classifier = RandomForestClassifier(n_jobs=-1, random_state=42)
rf_classifier.fit(X_train, y_train)
importances = rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
```

The cumulative importance is plotted to determine the optimal number of features:

```python
sorted_indices = np.argsort(importances)[::-1]
cumulative_importance = np.cumsum(importances[sorted_indices])

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(importances) + 1), cumulative_importance, 'b-')
```

### Multicollinearity Check

Variance Inflation Factor (VIF) is calculated to identify multicollinearity:

```python
vif = pd.DataFrame()
vif["Variable"] = X_train_for_vif.columns
vif["VIF"] = [variance_inflation_factor(X_train_for_vif.values, i) for i in range(X_train_for_vif.shape[1])]
```

Features with high VIF values may be redundant and candidates for removal.

## Model Building

### Traditional Machine Learning Models

The code implements and evaluates nine machine learning algorithms:

```python
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
```

Each algorithm is evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Building time

### Feature-Reduced Models

Based on feature importance, a reduced set of features is selected:

```python
X_train_select = X_train[['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']]
X_test_select = X_test[['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']]
```

The same nine algorithms are then evaluated using only these selected features.

### Neural Network Models

Five different neural network architectures are implemented and compared:

1. **Architecture I**: Simple three-layer network
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Dense(128, activation='relu', input_dim=X_train_select.shape[1]),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(1, activation='sigmoid')
   ])
   ```

2. **Architecture II**: Multi-layer network with dropout
   ```python
   model = Sequential()
   model.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=X_train_select.shape[1]))
   model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
   model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
   model.add(Dropout(0.25))
   model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
   ```

3. **Architecture III**: Smaller network with dropout
4. **Architecture IV**: Network with batch normalization
5. **Architecture V**: Network with higher dropout rates

Each architecture uses early stopping and model checkpointing to prevent overfitting:

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)
checkpoint = ModelCheckpoint('ChurnClassifier.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
```

### Ensemble Models

Three ensemble methods are implemented and compared:

```python
base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
adaboost_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
gradient_boost_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
xgboost_model = XGBClassifier(n_estimators=50, random_state=42)
```

### PCA Models

Principal Component Analysis (PCA) is applied to reduce dimensionality:

```python
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
```

The optimal number of components is determined using the scree plot:

```python
n_components = 8  # Based on scree plot analysis
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

The machine learning algorithms are then applied to the PCA-transformed data.

## Model Evaluation

### Metrics Comparison

All models are evaluated using multiple metrics:

```python
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
```

The results are compiled into DataFrames for easy comparison:

```python
df_test = pd.DataFrame(results_test)
print("\nTest metrics:")
print(df_test)
```

### ROC Curve Analysis

The ROC curve and AUC score are calculated for the best-performing model:

```python
y_test_prob = best_model.predict_proba(X_test_select)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
auc = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {auc:.2f})')
```

### Confusion Matrix

A confusion matrix is generated to visualize true positives, false positives, true negatives, and false negatives:

```python
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

## Hyperparameter Tuning

Grid search is applied to find optimal hyperparameters for the XGBoost model:

```python
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
```

## Final Model

### Model Performance

The final XGBoost model with optimized hyperparameters is evaluated:

```python
final_model = XGBClassifier(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_select, y_train)

y_test_pred = final_model.predict(X_test_select)
print("\nFinal model performance on test set:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
```

### Feature Importance

The importance of each feature in the final model is analyzed:

```python
feature_importances = final_model.feature_importances_
feature_names = X_train_select.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
```

### Cross-Validation

Cross-validation is performed to ensure the model's robustness:

```python
cv_scores = cross_val_score(final_model, X_train_select, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
```

## Model Deployment

### Prediction Function

A function is created to make predictions for new customers:

```python
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
    # For prediction we only need the features that were selected for the final model
    selected_features = ['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']
    
    # Make sure all required columns exist
    for feature in selected_features:
        if feature not in customer_data.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Select only the relevant features for prediction
    customer_features = customer_data[selected_features]
    
    # Make prediction directly with the selected features
    prediction = model.predict(customer_features)
    probability = model.predict_proba(customer_features)[:, 1]
    
    return prediction[0], probability[0]
```

An example demonstrates how to use the function:

```python
new_customer = pd.DataFrame({
    'Age': [35],
    'Monthly_Bill': [70.5],
    'Total_Usage_GB': [50],
    'Subscription_Length_Months': [12]
})

prediction, probability = predict_customer_churn(new_customer, loaded_model, None)
```

### Model Persistence

The final model and scaler are saved to disk for future use:

```python
joblib.dump(final_model, 'final_xgboost_model.pkl')
joblib.dump(scaler, 'scaler_model.pkl')
```

These can be loaded when needed:

```python
loaded_model = joblib.load('final_xgboost_model.pkl')
```

## Customer Segmentation

### Risk Segments

Customers are segmented based on their churn probability:

```python
def assign_segment(prob):
    if prob < 0.3:
        return 'Low Risk'
    elif prob < 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'

customer_segments['Risk_Segment'] = customer_segments['Churn_Probability'].apply(assign_segment)
```

### Segment Analysis

Statistical analysis is performed for each segment:

```python
segment_stats = customer_segments.groupby('Risk_Segment').agg({
    'Monthly_Bill': 'mean',
    'Total_Usage_GB': 'mean',
    'Age': 'mean',
    'Subscription_Length_Months': 'mean',
    'Churn_Probability': 'mean',
    'Actual_Churn': 'mean'
}).round(2)
```

The actual churn rate by segment is calculated:

```python
segment_churn_rates = customer_segments.groupby('Risk_Segment')['Actual_Churn'].mean().sort_values(ascending=False)
```

### Revenue Impact

The potential revenue at risk is calculated for each segment:

```python
customer_segments['Monthly_Revenue'] = X_test['Monthly_Bill']
revenue_at_risk = customer_segments.groupby('Risk_Segment').agg({
    'Monthly_Revenue': 'sum',
    'Churn_Probability': 'mean'
})
revenue_at_risk['Expected_Revenue_Loss'] = revenue_at_risk['Monthly_Revenue'] * revenue_at_risk['Churn_Probability']
```

## Business Recommendations

Based on the analysis, several business recommendations are provided:

1. **Focus retention efforts on high-risk segments**:
   ```
   Focus retention efforts on the High Risk segment which represents approximately
   X% of customers but accounts for $Y in monthly expected revenue loss.
   ```

2. **Analyze key churn factors**:
   ```
   Analyze the key factors driving churn in the High Risk segment:
   - Monthly_Bill: High Risk avg = X, Low Risk avg = Y
   - Total_Usage_GB: High Risk avg = X, Low Risk avg = Y
   - Age: High Risk avg = X, Low Risk avg = Y
   ```

3. **Implement targeted retention strategies**:
   ```
   Implement targeted retention strategies:
   - Offer discounts or loyalty rewards to high-risk customers
   - Provide personalized usage recommendations based on customer patterns
   - Implement proactive customer service outreach for at-risk customers
   ```

4. **Monitor medium-risk customers** to prevent migration to high-risk

5. **Regularly retrain the model** to adapt to changing customer behaviors
