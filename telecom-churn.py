#Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.simplefilter('ignore')


#1. Get the data

telcom = pd.read_csv(r"C:\Users\sandeep\Desktop\new\Telecom\Telco-Customer-Churn.csv")
#first 5 rows
telcom.head()


#2. Initial Analysis

telcom.info()

# convert all TotalCharges to numeric and set the invalid parsings/errors as NaN
telcom['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'], errors = 'coerce')

# check the rows which have NaN in the TotalCharges column
telcom.loc[telcom['TotalCharges'].isna()==True]

# we get 11 rows in the result , all of which have tenure =0
# Decision making - We can either drop these rows (as they are useless) or set the TotalCharges values to be 0. 
# Lets drop the rows.
#dataset = dataset.dropna()
telcom = telcom[telcom["TotalCharges"].notnull()]
telcom = telcom.reset_index()[telcom.columns]

# Converting SeniorCitizen from int to categorical 
telcom['SeniorCitizen']=pd.Categorical(telcom['SeniorCitizen'])

# Deleting the custumerID column, contains no useful information. And we already have the pandas numerical index.
telcom.drop(['customerID'],axis=1, inplace=True)
telcom.describe()

# split the dataset into numeric and objects to facilitate the analysis:

numerics = ['float64', 'int64']
numeric_ds = telcom.select_dtypes(include=numerics)
objects_ds = telcom.select_dtypes(exclude=numerics)

numeric_ds.describe()
objects_ds.describe().T
telcom.groupby('Churn').size()


#3. Graphical Analysis
#Target variable
#We will look into our target variable distribution by using categorical plot function of seaborn

ax = sns.catplot(y="Churn", kind="count", data=telcom, height=2.6, aspect=2.5, orient='h')



#We have a slightly unbalanced target: Churn: No - 72.4% Churn: Yes - 27.6%
#Numerical variables

#In this part we will look into our numerical variables, how they are distributed, how they relate with each other and how they can help us to predict the ‘Churn’ variable.

#There are only three numerical columns: tenure, monthly charges and total charges. The probability density distribution can be estimate using the seaborn kdeplot function.

def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(telcom[telcom['Churn'] == 'No'][feature].dropna(), color= 'navy', label= 'Churn: No')
    ax1 = sns.kdeplot(telcom[telcom['Churn'] == 'Yes'][feature].dropna(), color= 'orange', label= 'Churn: Yes')
kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')



#From the plots above we can conclude that:

  #  Recent clients are more likely to churn 
  # Clients with higher MonthlyCharges are also more likely to churn
  #  Tenure and MonthlyCharges are probably important features .

  # We can come to the same conclusions when we use scatter plots
g = sns.PairGrid(telcom, y_vars=["tenure"], x_vars=["MonthlyCharges", "TotalCharges"], height=4.5, hue="Churn", aspect=1.1)
ax = g.map(plt.scatter, alpha=0.6)


#Categorical variables

#This dataset has 16 categorical features:

    # Six binary features (Yes/No) 
    # Nine features with three unique values each (categories) 
    # One feature with four unique values 
    
      # Partner and dependents

fig, axis = plt.subplots(1, 2, figsize=(12,4))
axis[0].set_title("Has partner")
axis[1].set_title("Has dependents")
axis_y = "percentage of customers"
# Plot Partner column
gp_partner = telcom.groupby('Partner')["Churn"].value_counts()/len(telcom)
gp_partner = gp_partner.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()
ax = sns.barplot(x='Partner', y= axis_y, hue='Churn', data=gp_partner, ax=axis[0])
# Plot Dependents column
gp_dep = telcom.groupby('Dependents')["Churn"].value_counts()/len(telcom)
gp_dep = gp_dep.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()
ax = sns.barplot(x='Dependents', y= axis_y, hue='Churn', data=gp_dep, ax=axis[1])



#From the plots above we can conclude that: 
# Customers that doesn't have partners are more likely to churn 
# Customers without dependents are also more likely to churn

#Contract and Payment

# Adding a column here which is the answer , and which was inadvertently going into training as well as test data. 
#So accuracy was coming to 100% . Nice point for future teaching and assignments

#telcom['churn_rate'] = telcom['Churn'].replace("No", 0).replace("Yes", 1)
#g = sns.FacetGrid(telcom, col="PaperlessBilling", height=4, aspect=.9)
#ax = g.map(sns.barplot, "Contract", "churn_rate", palette = "Blues_d", order= ['Month-to-month', 'One year', 'Two year'])



def barplot_percentages(feature, orient='v', axis_name="percentage of customers"):
    ratios = pd.DataFrame()
    g = telcom.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(telcom)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()

plt.figure(figsize=(9, 4.5))
barplot_percentages("PaymentMethod", orient='h')

# A few observations:
# Customers with paperless billing are more probable to churn
# The preferred payment method is Electronic check with around 35% of customers. This method also has a very high churn rate
# Short term contracts have higher churn rates
# One and two year contracts probably have contractual fines and therefore customers have to wait untill the end of contract to churn.
# These observations are important for when we design the retention campaigns so that we know where we can focus.

# plot all categorical variables as a bar plot
fig,ax =plt.subplots(4,4,figsize=(15,15))
fig.subplots_adjust(hspace=.5)
for i in range(0,16):
    g = sns.countplot(x=objects_ds.iloc[:,i], hue=objects_ds["Churn"], ax=ax[divmod(i,4)])
    g.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.) if i==0 else g.legend_.remove()
for tick in ax[3,3].get_xticklabels():
    tick.set_rotation(45)

#Now we have a better picture of the variables that are more important to us, 
# for example, having Month-to-month contract is a strong indicator if the client might leave soon, 
# so is the Electronic check payment method, being a senior citizen on the other hand is a good predictor but only represents a small amount of the companies clients
#  so you might prefer to focus on the variables that delivers the best results first before tackling it.



#4. Machine Learning Models And Performance Evaluation

#We will use Logistic Regression, Decision Tree, Random Forest and SVM. First set aside a test data.
#If we don't encode the categorical variables into numeric, the model is throwing error.

# categorical variable encoding
cat_vars_list = objects_ds.columns.tolist()
## Label Encoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in cat_vars_list :
    telcom[i] = le.fit_transform(telcom[i])
telcom


#Divide dataset into training and test dataset

## divide dataset into train and test datasets

from sklearn.model_selection import train_test_split

X = telcom.drop(columns = ['Churn'])
Y = telcom['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=990143)

#Logistic Regression

from sklearn.linear_model import LogisticRegression

logisticReg = LogisticRegression()
result = logisticReg.fit(X_train, Y_train)
Y_pred_lr = logisticReg.predict(X_test)
print (Y_pred_lr)


from sklearn import metrics

print (metrics.accuracy_score(Y_test, Y_pred_lr))

# what is the accuracy on training data. Is it overfit?
Y_train_predict_lr = logisticReg.predict(X_train)
print (metrics.accuracy_score(Y_train, Y_train_predict_lr))

#Confusion Matrix and Precision, Recall

#Define a function to print the confusion matrix, precision and recall for a given model

## confusion matrix and precision recall
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score


def cm_and_pr(Y_pred, model_name):
    cm = confusion_matrix(Y_test, Y_pred)
    df_cm = pd.DataFrame(cm)
    ax= plt.subplot()

    classNames = ['No', 'Yes']

    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True, xticklabels=classNames, yticklabels=classNames, annot_kws={"size": 16}, fmt='g')

    ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
    ax.set_title('Confusion Matrix'); 
         
    #precision and recall
    
    # true positive - churn, predicted as 'churn'
    # true negative - no churn, predicted as 'no churn'
    # false positive - no churn, predicted as 'churn'
    # false negative - churn, predicted as 'no churn'
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    precision = precision_score(Y_test, Y_pred, labels = [1], average='micro')
    recallscore = recall_score(Y_test, Y_pred, labels = [1], average='micro')

    data = [[model_name, precision, recallscore]] 


    df = pd.DataFrame(data, columns = ['Model', 'Precision', 'Recall'])
    print(df)    

## logistic regression
cm_and_pr(Y_pred_lr, "Logistic Regression")

## 2 variable logistic regression

X_train1 = X_train[['tenure', 'MonthlyCharges']]  # we only take the first two features.

logisticReg1 = LogisticRegression()
result = logisticReg1.fit(X_train1, Y_train)

X_test1 = X_test[['tenure', 'MonthlyCharges']]

Y_test_pred_lr1 = logisticReg1.predict(X_test1)
print (metrics.accuracy_score(Y_test, Y_test_pred_lr1))


#Decision Tree

## decision tree 
from sklearn import tree, metrics

dec_tree = tree.DecisionTreeClassifier()
result_tree = dec_tree.fit(X_train, Y_train)
Y_pred_dt = dec_tree.predict(X_test)
print (metrics.accuracy_score(Y_test, Y_pred_dt))

# tree.score same as metrics.accuracy_score
print (dec_tree.score(X_test, Y_test))

# Checking overfitting on the training data
Y_train_pred_dt = dec_tree.predict(X_train)
print (metrics.accuracy_score(Y_train, Y_train_pred_dt))

## (Generalization) Pruning of tree  - put upper cap on decision tree depth
dec_tree1 = tree.DecisionTreeClassifier(max_depth = 4)
result_tree1 = dec_tree1.fit(X_train, Y_train)
Y_test_pred_dt1 = dec_tree1.predict(X_test)
print (metrics.accuracy_score(Y_test, Y_test_pred_dt1))

print ("depth 4 training accuracy - ", dec_tree1.score(X_train, Y_train))

cm_and_pr(Y_test_pred_dt1, "Decision Tree")


#Visualizing decision tree

# dot_data = tree.export_graphviz(dec_tree1, out_file=None, 
#                                 feature_names=X.columns,  
# #                                 class_names=["No", "Yes"])

# Draw graph
## ** conda install graphviz python-graphviz
## conda install pydotplus

# import pydotplus
# from IPython.display import Image

# graph = pydotplus.graph_from_dot_data(dot_data)  

# # Show graph
# Image(graph.create_png())

#Random Forest

from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier()
randomForest.fit(X_train, Y_train)
Y_pred_rf = randomForest.predict(X_test)

print (randomForest.score(X_test, Y_test))

# Checking overfitting on the training data
print (randomForest.score(X_train, Y_train))

# Random Forest - fixing overfitting
randomForest1 = RandomForestClassifier(n_estimators=4, max_depth=4) #by default is 10 (estimators are trees).
randomForest1.fit(X_train, Y_train)
Y_pred_rf1 = randomForest1.predict(X_test)

print (randomForest1.score(X_test, Y_test))

# Checking overfitting on the training data
print (randomForest1.score(X_train, Y_train))

cm_and_pr(Y_pred_rf1, "Random Forest")


#SVM

from sklearn.svm import SVC

svc_cl = SVC()
svc_cl.fit(X_train, Y_train)
Y_pred_svm = svc_cl.predict(X_test)

# SVM
from sklearn.svm import SVC

svc_cl = SVC()
svc_cl.fit(X_train, Y_train)
Y_pred_svm = svc_cl.predict(X_test)

print (svc_cl.score(X_test, Y_test))
print (svc_cl.score(X_train, Y_train))

cm_and_pr(Y_pred_svm, "SVM")


























