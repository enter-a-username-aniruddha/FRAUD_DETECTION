#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading the dataset

# In[2]:


data = pd.read_csv('Fraud.csv')


# In[3]:


data.head()


# ## Dropping irrelevant columns

# In[4]:


data = data.drop(['nameOrig', 'nameDest', 'step'], axis=1)


# ## Checking for missing values

# In[5]:


data.isna().sum()


# ## Relation between the vairables

# In[6]:


correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Oranges')
plt.title('Correlation Matrix')
plt.show()


# ### Since it was found that there exist correlation between explanatory variables this implies the presence of multicollinearity. Now we will check the variance inflation factor for each numerical variable in the data

# In[7]:


# VIF Calculation
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data


# In[8]:


numerical_features = data.select_dtypes(include=['float64', 'int64'])


# In[9]:


vif_df = calculate_vif(numerical_features)

# Display VIF values
print("VIF Values:")
print(vif_df)


# ### As we can see that oldbalanceOrg, newbalanceOrig,  oldbalanceDest, newbalanceDest  are highly correlated with each other

# ###  One hot encoding the variable "type" and separating the data into x and y

# In[10]:


data = pd.get_dummies(data, columns=['type'], drop_first=True)
X = data.drop(columns=['isFraud'], axis=1)
y = data['isFraud']


# ### Scaling the values using Standard Scaler

# In[12]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)


# ### As mentioned above there is a presence of multicollinearity so we will use Principal Component Analysis (PCA) to deal with it it reduces the dimensions as well as it reduces the multicollinearity

# In[14]:


pca = PCA(n_components=0.95)  # Retain 95% of variance
x_pca = pca.fit_transform(x_scaled)


# In[15]:


data.head()


# In[16]:


x_pca


# ### Splitting the data into training and testing sets

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)


# ### XGBoost (Extreme Gradient Boosting) used for classification problem and it is suitable for large datasets

# In[18]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)


# In[20]:


print("Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ### This is the report of the model as we can see that the accuracy score is 0.9995 which means the model has an accuracy of 99.95% which clearly says overfitting , this can happen due to disproportion between the values in the Dependent variable 

# In[21]:


data['isFraud'].value_counts()


# ### We can clearly see that the count of 0's are far much dominating than the count of 1's which makes our model a biased one and will perform excellently on seen data but it will perform poorly on unseen data

# In[ ]:





# ## Alternative approach using Logistic regression

# ### to deal with imbalance we will assign weights for 0 & 1 in the 'Y' variable.

# ### For the majority class (class 0), the weight is set to 1. This means that the misclassification of a sample from class 0 is treated with a weight of 1. For the minority class (class 1), the weight is set to 10. This implies that the misclassification of a sample from class 1 is treated with a weight of 10. This is often done to give more importance to the minority class in situations where the dataset is imbalanced.

# In[22]:


from sklearn.linear_model import LogisticRegression
class_weight = {0: 1, 1: 10}  # Adjust the weights based on the imbalance
logistic_model = LogisticRegression(class_weight=class_weight, random_state=42)
logistic_model.fit(X_train, y_train)


# In[23]:


y_pred_logistic_weighted = logistic_model.predict(X_test)


# In[24]:


print("Logistic Regression Model Evaluation with Class Weights:")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic_weighted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic_weighted))
print("Classification Report:")
print(classification_report(y_test, y_pred_logistic_weighted))


# ### Even after assigning weights there's a problem of overfitting

# ### To tackle this probelm we will use SMOTE , (Synthetic Minority Over-sampling Technique) is a specific algorithm for oversampling the minority class in imbalanced datasets. It is often used to tackle the class imbalance problem , especially in binary classification tasks.

# ### SMOTE is used to oversample the minority class (class 1) in the training data, and the sampling_strategy parameter is set to 0.5, meaning the number of synthetic samples for the minority class will be 50% of the number of samples in the majority class.

# In[26]:


from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train Logistic Regression on the resampled data
logistic_model_resampled = LogisticRegression(class_weight=class_weight, random_state=42)
logistic_model_resampled.fit(X_resampled, y_resampled)

# Evaluate the model on the test set
y_pred_resampled = logistic_model_resampled.predict(X_test)


# In[27]:


print("Logistic Regression Model Evaluation with SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred_resampled))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_resampled))
print("Classification Report:")
print(classification_report(y_test, y_pred_resampled))


# ### This is the report of the model after SMOTE the accuracy of the model is 76.12% which is not as desired as we want

# In[28]:


fpr_resampled, tpr_resampled, thresholds_roc_resampled = roc_curve(y_test, y_pred_resampled )
roc_auc_resampled = auc(fpr_resampled, tpr_resampled)

plt.figure(figsize=(12, 6))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr_resampled, tpr_resampled, label=f'ROC Curve (AUC = {roc_auc_resampled:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve (Resampled)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()


# In[ ]:





# ### Optimizing the accuracy 
# ### GridSearchCV Used for hyperparameter tuning by performing an exhaustive search over a specified parameter grid. 
# ### The hyperparameter grid is a dictionary containing different values of the hyperparameter C (inverse of regularization strength) to be searched during the grid search.
# ###

# In[29]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

logistic_model = LogisticRegression(class_weight=class_weight, random_state=42)

grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_logistic_model = grid_search.best_estimator_

y_pred_best_model = best_logistic_model.predict(X_test)
accuracy_best_model = accuracy_score(y_test, y_pred_best_model)
print("Accuracy with Best Model:", accuracy_best_model)


# ### This is the optimized model with improved accuracy of 80.30%

# In[31]:


print("Accuracy with Best Model:", accuracy_best_model)
conf_matrix_best_model = confusion_matrix(y_test, y_pred_best_model)
print("Confusion Matrix:")
print(conf_matrix_best_model)

# Print Classification Report
class_report_best_model = classification_report(y_test, y_pred_best_model)
print("Classification Report:")
print(class_report_best_model)


# In[34]:


y_probs_best_model = best_logistic_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs_best_model)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ### Checking feature importance

# In[36]:


data.head()


# In[43]:


coefficients = best_logistic_model.coef_[0]
coefficients


# In[47]:


feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']


# In[48]:


coefficients = [776.13966241, 235.27287193, -1315.24591862, -1134.53775684, 198.92675182, 349.63919379, -483.84600543]
coefficients_dict = dict(zip(feature_names, coefficients))
for feature, coefficient in coefficients_dict.items():
    print(f"{feature}: {coefficient}")


# #### amount: A positive coefficient indicates that as the transaction amount increases, the log-odds of the transaction being fraudulent also increase.
# 
# #### oldbalanceOrg: A positive coefficient suggests that as the old balance of the origin account increases, the log-odds of the transaction being fraudulent also increase.
# 
# #### newbalanceOrig: A negative coefficient suggests that as the new balance of the origin account increases, the log-odds of the transaction being fraudulent decrease.
# 
# #### oldbalanceDest: A negative coefficient indicates that as the old balance of the destination account increases, the log-odds of the transaction being fraudulent decrease.
# 
# #### newbalanceDest: A positive coefficient suggests that as the new balance of the destination account increases, the log-odds of the transaction being fraudulent also increase.
# 
# #### isFraud: A positive coefficient for the target variable itself indicates that the presence of fraud in the transaction increases the log-odds of the transaction being fraudulent (which is expected).
# 
# #### isFlaggedFraud: A negative coefficient suggests that the presence of a flagged fraud decreases the log-odds of the transaction being fraudulent.

# In[ ]:





# ### Prevention against fraud

# ### Educate customers about common fraud schemes and best practices for securing their accounts.Encourage customers to regularly monitor their accounts for any unusual activity
# 
# ### Provide regular training sessions for employees to stay informed about the latest fraud trends and prevention techniques. Ensure that staff members are equipped to identify and respond to potential fraud incidents.
# 
# ### Implement transaction limits and controls to restrict the amount and frequency of transactions
# 
# ### Analyze user behavior and transaction patterns to identify anamoly from normal behavior.
# 
