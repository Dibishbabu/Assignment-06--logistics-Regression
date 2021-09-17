#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[9]:


#Importing the dataset
data = pd.read_csv('D:/Assignment-06-Logistic Regression/bank-full.csv', sep =";")
data.head()


# In[10]:


data.tail()


# In[11]:


data.info()


# In[12]:


data1= data.replace(to_replace='none', value=np.nan)


# In[13]:


data1.info()


# In[14]:


data2 = data1.dropna()


# In[15]:


data2.shape


# ### 

# In[16]:


#Count of duplicated rows
data2[data2.duplicated()].shape


# In[17]:


# # Custom Binary Encoding of Binary o/p variables can be used if not use category for all
# data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
# data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
# data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
# data1['y'] = np.where(data1['y'].str.contains("yes"), 1, 0)
# data1


# In[18]:


#Creating dummy variable for Weather column
data3=pd.get_dummies(data2,columns=['job', 'month','marital','education','default', 'housing', 'loan', 'contact', 'poutcome', 'y'])


# In[19]:


data3.head()


# In[20]:


# To see all columns
pd.set_option("display.max.columns", None)
data3


# In[21]:


#dropping the case number columns as it is not required
data3.drop(['y_no'],inplace=True,axis = 1)


# In[22]:


data3.head()


# In[23]:


# Dividing our data into input and output variables 
X = data3.iloc[:,0:-1]
Y = data3.iloc[:,-1]


# In[24]:


X.head()


# In[25]:


Y.head()


# In[26]:


#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(X,Y)


# In[27]:


#Predict for X dataset
y_pred = classifier.predict(X)


# In[28]:


y_pred


# In[29]:



y_pred_df= pd.DataFrame({'actual': Y,
                         'predicted_prob': classifier.predict(X)})


# In[30]:


y_pred_df


# In[31]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[32]:


a=39119
b=803
c=4118
d=1171


# In[33]:


accuracy= ((a+d)/(a+b+c+d))*100
print("Overall Accuracy is : {}". format(accuracy))


# # precision= ((a)/(a+b))*100
# print("Precision is : {}". format(precision))

# In[35]:


sensitivity =((a)/(a+c))*100   #Recall
print("Sensitivity is : {}". format(sensitivity))


# In[36]:


Specificity= ((d)/(b+d))*100
print("Overall specificity is : {}". format(Specificity))


# In[37]:


F1_score = 2*sensitivity*precision/(sensitivity+precision)


# In[38]:


print("F1 score is :{} %".format(F1_score))


# # We devloped a model with overall accuaracy 89.1%, precison is 97.98%, specificity (recall) is 59.32%, sensitivity is 90.47%. F1 score is 0.94 and it is very close to 1. So the model is good. But yes, there is scope to improve model because the sensitivity is higher, but there is scope to improve recall by reducing the true positive.

# In[ ]:


# ROC Curve


# In[40]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[41]:


auc # Area under the red curve


# # Total area under ROC curve is 0.60. So there is scope to improve the model by reducing the false negative and false positive values
