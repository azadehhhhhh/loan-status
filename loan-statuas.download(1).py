#!/usr/bin/env python
# coding: utf-8

# In[ ]:





#  # 1-Summary

# ## Data Set Problems 🤔
# 
# 👉 The company seeks **to automate (in real time) the loan qualifying procedure** based on information given by customers while filling out an online application form. It is expected that the development of ML models that can help the company predict loan approval in **accelerating decision-making process** for determining whether an applicant is eligible for a loan or not.

# 👨‍💻 **The machine learning models used in this project are:** 
# 1. Logistic Regression
# 2. K-Nearest Neighbour (KNN)
# 3. Support Vector Machine (SVM)
# 4. Naive Bayes
# 5. Decision Tree
# 6. Random Forest
# 7. Gradient Boost
# 

# 👨‍💻 **The machine learning models used in this project are:** 
# 1. Logistic Regression
# 2. K-Nearest Neighbour (KNN)
# 3. Support Vector Machine (SVM)
# 4. Naive Bayes
# 5. Decision Tree
# 6. Random Forest
# 7. Gradient Boost

# ## Data Set Description 🧾
# 👉 There are **13 variables** in this data set:
# *   **8 categorical** variables,
# *   **4 continuous** variables, and
# *   **1** variable to accommodate the loan ID.
# 

# # 2. Importing Libraries 📚
# 👉 Importing libraries that will be used in this notebook.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import missingno as mso
import seaborn as sns
import warnings
import os
import scipy
from sklearn.preprocessing import LabelEncoder

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score


# # 3. Reading Data Set

# In[2]:


df = pd.read_csv("../input/loan-data/train.csv")


# In[3]:


df.head()


# In[4]:


plt.figure(figsize=(10, 3))
sns.barplot(x=df['Loan_Status'].value_counts().index, y=df['Loan_Status'].value_counts(), palette='rainbow')
plt.xlabel('Loan_Status')
plt.ylabel('Count')
plt.show()


# In[5]:


#df2 = pd.read_csv("../input/loan-data/train.csv")


# In[6]:


df.shape


# In[7]:


df.info()


# df[['Gender', 'Married', 'Dependents', 'Self_Employed']] = df[['Gender', 'Married', 'Dependents', 'Self_Employed']].fillna(df[['Gender', 'Married', 'Dependents', 'Self_Employed']].mode().iloc[0])# 3. Data prepration 

# # 4. Data Cleaning

# # 4.1 Fill Null Values
# 

# In[8]:


# Count the null values in each column
null_counts = df.isnull().sum()

# Print the null counts
print(null_counts)


# In[9]:


discrete_columns=['Gender', 'Married', 'Dependents', 'Self_Employed']
df[discrete_columns] = df[discrete_columns].fillna(df[discrete_columns].mode().iloc[0])


# In[10]:


# Count the null values in each column
null_counts = df.isnull().sum()

# Print the null counts
print(null_counts)


# In[11]:


# Create an instance of KNNImputer with the desired number of neighbors
imputer = KNNImputer(n_neighbors=5)

# Select the continuous columns
continuous_columns = ['Credit_History', 'LoanAmount', 'Loan_Amount_Term']

# Fill null values using KNN imputation
df[continuous_columns] = imputer.fit_transform(df[continuous_columns])


# In[12]:


# Count the null values in each column
null_counts = df.isnull().sum()

# Print the null counts
print(null_counts)


# In[13]:


df = df.drop('Loan_ID', axis=1)


# In[14]:


df.head()


# # 4.2 Data Encoding

# In[15]:


categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Loan_Status']
X_encoded = pd.get_dummies(df, columns=categorical_columns)


# In[16]:


X_encoded.head()


# In[17]:


X_encoded.shape


# In[18]:


categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df_train = pd.get_dummies(df, columns=categorical_columns)


# In[19]:


df_train.head()


# In[20]:


# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1

# df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[21]:


# Select the numeric variables
#numeric_vars = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Normalize the numeric variables
#scaler = MinMaxScaler()
#df[numeric_vars] = scaler.fit_transform(df[numeric_vars])

# Print the normalized data
#print(df)


# In[22]:


df_train.shape


# # 5. Visualization
# 

# # 5.1 Correlation Plot
# 

# In[23]:


# Create a correlation matrix
cor = X_encoded.corr()

# Create a heatmap with annotations
plt.figure(figsize=(16, 16))
sns.heatmap(cor, annot=True, fmt=".2f", cmap='Set1')

# Add a title
plt.title('Correlation Heatmap with Values')

# Display the plot
plt.show()


# In[24]:


# Numeric Variables
numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
# Categorical Variables
categorical_columns =['Gender_Female',
       'Gender_Male', 'Married_No', 'Married_Yes', 'Dependents_0',
       'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate',
       'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
       'Property_Area_Rural', 'Property_Area_Semiurban',
       'Property_Area_Urban'] 


# In[25]:


for feature in numeric_columns:
    # Create a histogram for the numeric feature by Loan_Status
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_train, x=feature, hue='Loan_Status', kde=True)
    plt.title(f'{feature} Histogram by Loan Status')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Loan_Status', labels=['Not Approved (0)', 'Approved (1)'])
    plt.show()


# In[26]:


df_train.columns


# In[ ]:





# In[27]:


for feature in categorical_columns:
    # Create a countplot for the categorical feature by Loan_Status
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_train, x=feature, hue='Loan_Status')
    plt.title(f'{feature} Countplot by Loan Status')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Loan_Status', labels=['Not Approved (0)', 'Approved (1)'])
    plt.xticks(rotation=45)
    plt.show()


# # 6. Outlier Detection

# In[28]:


# Set a Z-score threshold for identifying outliers
z_threshold = 3.0

# Create a dictionary to store the number of outliers for each column
outliers_count = {}

for feature in numeric_columns:
    # Calculate the Z-score for each data point in the feature
    z_scores = stats.zscore(df_train[feature])
    
    # Find indices of data points with Z-scores exceeding the threshold
    outlier_indices = abs(z_scores) > z_threshold
    
    # Count the number of outliers
    num_outliers = sum(outlier_indices)
    
    # Store the number of outliers in the dictionary
    outliers_count[feature] = num_outliers

# Print the number of outliers for each column
for feature, count in outliers_count.items():
    print(f"Number of outliers in {feature}: {count}")


# ## 7. Splitting Data Set 🪓

# In[29]:


# Assuming 'data' is your DataFrame that contains the features and the target variable
X = df_train.drop('Loan_Status', axis=1)
y = df_train['Loan_Status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


from sklearn.preprocessing import StandardScaler

#Create the StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training data and transform X_train
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled= scaler.fit_transform(X_test)
# Create a new DataFrame with the scaled values and the original column names
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# X_train_scaled_df is now a DataFrame with scaled values


# In[31]:


X_train.head(3)


# # 6. Models 🛠

# In[32]:


# Create an instance of Logistic Regression
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[33]:


# Create an instance of KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[34]:


# Create an instance of SVC
svm = SVC(kernel='linear')

# Train the SVM model on the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[35]:


X[X < 0] = 0

# Encode categorical variables if present
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create the CategoricalNB model
nb = CategoricalNB()

# Fit the model on the training data
nb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb.predict(X_test)

# Calculate the accuracy of the model
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)






# In[36]:


# Create an instance of GaussianNB
nb = GaussianNB()

# Train the GaussianNB model on the training data
nb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[37]:


# Create an instance of DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Train the Decision Tree model on the training data
dt.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[38]:


# Create an instance of RandomForestClassifier
rf = RandomForestClassifier()

# Train the Random Forest model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[39]:


# Create an instance of GradientBoostingClassifier
gb = GradientBoostingClassifier()

# Train the Gradient Boosting model on the training data
gb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gb.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ## 7.saving trained model

# In[40]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
filename = 'trained_model.sav'
pickle.dump(gb, open(filename,'wb'))
#load saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))


# In[41]:


#load saved model
input_data=(441,0,155,11,5,0,1,0,1,1,0,0,0,1,0,1,0,0,1,0)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
    print('approved')
else:
    print('Not approved')


# In[42]:


X_test.columns
      


# In[43]:


X_test.head(3)


# In[ ]:




