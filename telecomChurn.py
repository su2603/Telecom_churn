# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.simplefilter('ignore')

# 1. Get the data
def load_data():
    telcom = pd.read_csv("Telco-Customer-Churn.csv")
    return telcom

telcom = load_data()
# Display first 5 rows
print(telcom.head())

# 2. Initial Analysis
# Display information about the dataframe
print(telcom.info())

# Convert all TotalCharges to numeric and set the invalid parsings/errors as NaN
telcom['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'], errors='coerce')

# Check the rows which have NaN in the TotalCharges column
print("Rows with NaN in TotalCharges:")
print(telcom.loc[telcom['TotalCharges'].isna()==True])

# We get 11 rows in the result, all of which have tenure = 0
# Decision: Drop these rows as they appear to be new customers with no charges yet
telcom = telcom[telcom["TotalCharges"].notnull()]
telcom = telcom.reset_index()[telcom.columns]

# Converting SeniorCitizen from int to categorical 
telcom['SeniorCitizen'] = telcom['SeniorCitizen'].astype('category')

# Deleting the customerID column, contains no useful information
telcom.drop(['customerID'], axis=1, inplace=True)

# Show basic statistics of the dataset
print(telcom.describe())

# Split the dataset into numeric and objects to facilitate the analysis:
numerics = ['float64', 'int64']
numeric_ds = telcom.select_dtypes(include=numerics)
objects_ds = telcom.select_dtypes(exclude=numerics)

print(numeric_ds.describe())
print(objects_ds.describe().T)

# Show distribution of target variable
print(telcom.groupby('Churn').size())

# 3. Graphical Analysis
# Target variable visualization
plt.figure(figsize=(10, 5))
ax = sns.countplot(y="Churn", data=telcom)
plt.title("Distribution of Churn")
plt.show()

# Function for KDE plots
def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title(f"KDE for {feature}")
    sns.kdeplot(telcom[telcom['Churn'] == 'No'][feature].dropna(), color='navy', label='Churn: No')
    sns.kdeplot(telcom[telcom['Churn'] == 'Yes'][feature].dropna(), color='orange', label='Churn: Yes')
    plt.legend()
    plt.show()

# Plot KDE for numerical features
kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')

# Scatter plots for numerical variables
plt.figure(figsize=(12, 5))
g = sns.PairGrid(telcom, y_vars=["tenure"], x_vars=["MonthlyCharges", "TotalCharges"], 
                height=4.5, hue="Churn", aspect=1.1)
g.map(plt.scatter, alpha=0.6)
plt.show()

# Categorical variables: Partner and Dependents
fig, axis = plt.subplots(1, 2, figsize=(12, 4))
axis[0].set_title("Has partner")
axis[1].set_title("Has dependents")

# Plot Partner column - FIXED VERSION
partner_churn = telcom.groupby(['Partner', 'Churn']).size().reset_index(name='count')
partner_churn['percentage'] = partner_churn['count'] / len(telcom)
sns.barplot(x='Partner', y='percentage', hue='Churn', data=partner_churn, ax=axis[0])
axis[0].set_ylabel("Percentage of customers")

# Plot Dependents column - FIXED VERSION
dep_churn = telcom.groupby(['Dependents', 'Churn']).size().reset_index(name='count')
dep_churn['percentage'] = dep_churn['count'] / len(telcom)
sns.barplot(x='Dependents', y='percentage', hue='Churn', data=dep_churn, ax=axis[1])
axis[1].set_ylabel("Percentage of customers")

plt.tight_layout()
plt.show()

# Function for creating percentage barplots - FIXED VERSION
def barplot_percentages(feature, orient='v'):
    plt.figure(figsize=(9, 4.5))
    
    # Create a dataframe with counts and percentages
    data = telcom.groupby([feature, 'Churn']).size().reset_index(name='count')
    data['percentage'] = data['count'] / len(telcom)
    
    if orient == 'v':
        # For vertical orientation
        ax = sns.barplot(x=feature, y='percentage', hue='Churn', data=data)
        ax.set_ylabel("Percentage of customers")
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        # For horizontal orientation
        ax = sns.barplot(y=feature, x='percentage', hue='Churn', data=data)
        ax.set_xlabel("Percentage of customers")
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    
    plt.title(f"Churn by {feature}")
    plt.tight_layout()
    plt.show()

# Plot payment method
barplot_percentages("PaymentMethod", orient='h')
barplot_percentages("Contract", orient='v')

# Plot all categorical variables
fig, ax = plt.subplots(4, 4, figsize=(15, 15))
fig.subplots_adjust(hspace=.5)

cat_cols = objects_ds.columns.tolist()
for i, col in enumerate(cat_cols):
    row, col_idx = divmod(i, 4)
    if i < 16:  # Make sure we don't go out of bounds
        g = sns.countplot(x=col, hue="Churn", data=telcom, ax=ax[row, col_idx])
        if i == 0:
            g.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                    mode="expand", borderaxespad=0.)
        else:
            g.legend_.remove()
        
        # Rotate x-tick labels if they might overlap
        if len(telcom[col].unique()) > 2:
            for tick in g.get_xticklabels():
                tick.set_rotation(45)

plt.tight_layout()
plt.show()

# 4. Machine Learning Models And Performance Evaluation

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in objects_ds.columns:
    telcom[col] = le.fit_transform(telcom[col])

# Divide dataset into training and test dataset
from sklearn.model_selection import train_test_split

X = telcom.drop(columns=['Churn'])
Y = telcom['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=990143)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticReg = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
result = logisticReg.fit(X_train, Y_train)
Y_pred_lr = logisticReg.predict(X_test)
print("Logistic Regression predictions:", Y_pred_lr)
print("Logistic Regression accuracy on test data:", metrics.accuracy_score(Y_test, Y_pred_lr))

# Check for overfitting
Y_train_predict_lr = logisticReg.predict(X_train)
print("Logistic Regression accuracy on training data:", metrics.accuracy_score(Y_train, Y_train_predict_lr))

# Function to print confusion matrix, precision and recall
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def cm_and_pr(Y_pred, model_name):
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(8, 6))
    
    class_names = ['No', 'Yes']
    
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, 
                annot_kws={"size": 16}, fmt='g')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()
    
    # Precision and recall calculations
    precision = precision_score(Y_test, Y_pred)
    recall_val = recall_score(Y_test, Y_pred)
    
    print(f"\n{model_name} Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall_val:.4f}")

# Visualize Logistic Regression results
cm_and_pr(Y_pred_lr, "Logistic Regression")

# 2-variable logistic regression
X_train_2var = X_train[['tenure', 'MonthlyCharges']]
logisticReg_2var = LogisticRegression(max_iter=1000)
logisticReg_2var.fit(X_train_2var, Y_train)

X_test_2var = X_test[['tenure', 'MonthlyCharges']]
Y_pred_lr_2var = logisticReg_2var.predict(X_test_2var)
print("2-variable Logistic Regression accuracy:", metrics.accuracy_score(Y_test, Y_pred_lr_2var))
cm_and_pr(Y_pred_lr_2var, "2-Variable Logistic Regression")

# Decision Tree 
from sklearn import tree

dec_tree = tree.DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, Y_train)
Y_pred_dt = dec_tree.predict(X_test)
print("Decision Tree accuracy on test data:", metrics.accuracy_score(Y_test, Y_pred_dt))

# Check for overfitting
Y_train_pred_dt = dec_tree.predict(X_train)
print("Decision Tree accuracy on training data:", metrics.accuracy_score(Y_train, Y_train_pred_dt))

# Pruning the tree - cap on decision tree depth
dec_tree_pruned = tree.DecisionTreeClassifier(max_depth=4, random_state=42)
dec_tree_pruned.fit(X_train, Y_train)
Y_pred_dt_pruned = dec_tree_pruned.predict(X_test)
print("Pruned Decision Tree (depth=4) accuracy on test data:", 
      metrics.accuracy_score(Y_test, Y_pred_dt_pruned))
print("Pruned Decision Tree (depth=4) accuracy on training data:", 
      dec_tree_pruned.score(X_train, Y_train))

cm_and_pr(Y_pred_dt_pruned, "Decision Tree (Pruned)")

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
print("Random Forest accuracy on test data:", rf.score(X_test, Y_test))
print("Random Forest accuracy on training data:", rf.score(X_train, Y_train))

# Random Forest - fixing overfitting
rf_tuned = RandomForestClassifier(n_estimators=4, max_depth=4, random_state=42)
rf_tuned.fit(X_train, Y_train)
Y_pred_rf_tuned = rf_tuned.predict(X_test)
print("Tuned Random Forest accuracy on test data:", rf_tuned.score(X_test, Y_test))
print("Tuned Random Forest accuracy on training data:", rf_tuned.score(X_train, Y_train))

cm_and_pr(Y_pred_rf_tuned, "Random Forest (Tuned)")

# SVM
from sklearn.svm import SVC

svc_model = SVC(random_state=42)
svc_model.fit(X_train, Y_train)
Y_pred_svm = svc_model.predict(X_test)

print("SVM accuracy on test data:", svc_model.score(X_test, Y_test))
print("SVM accuracy on training data:", svc_model.score(X_train, Y_train))

cm_and_pr(Y_pred_svm, "SVM")

# Feature importance visualization for Random Forest
plt.figure(figsize=(10, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

# Model comparison
models = ['Logistic Regression', '2-Var Logistic Regression', 'Decision Tree (Pruned)', 
          'Random Forest (Tuned)', 'SVM']
test_scores = [
    metrics.accuracy_score(Y_test, Y_pred_lr),
    metrics.accuracy_score(Y_test, Y_pred_lr_2var),
    metrics.accuracy_score(Y_test, Y_pred_dt_pruned),
    metrics.accuracy_score(Y_test, Y_pred_rf_tuned),
    metrics.accuracy_score(Y_test, Y_pred_svm)
]

plt.figure(figsize=(10, 6))
sns.barplot(x=test_scores, y=models)
plt.title('Model Comparison - Test Accuracy')
plt.xlabel('Accuracy')
plt.xlim(0.7, 0.85)  # Adjust as needed based on your results
plt.tight_layout()
plt.show()

print("Analysis complete!")