
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rm


# In[43]:


from sklearn import metrics


# In[2]:


# Setting the Directory
os.chdir("E:/Analytics/edWisor/Project/Santander/")


# In[3]:


# Importing the Database
SB_train = pd.read_csv("train.csv")


# In[4]:


SB_train.shape


# In[5]:


SB_train.dtypes


# In[6]:


#Converting target to categorical
SB_train['target'] = SB_train['target'].astype('category', copy = False)


# In[7]:


SB_train['target'].dtype


# In[8]:


plt.ylabel("Freq")
SB_train.groupby('target')['ID_code'].count().plot.bar()


# In[9]:


#Missing value analysis
miss_val = pd.DataFrame(SB_train.isnull().sum())
miss_val = miss_val.reset_index()
miss_val = miss_val.rename(columns = {'index': 'Variable_Names', 0: 'Missing_percent'})
miss_val['Missing_percent'] = (miss_val['Missing_percent']/len(SB_train))*100


# In[10]:


miss_val.sort_values('Missing_percent', ascending=False, inplace=True)


# In[11]:


miss_val.head(10)


# In[12]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[13]:


SB_pca = PCA()
SB_pca.fit(SB_train.iloc[:,2:201])


# In[14]:


#Cumulative explained variance ratio
SB_pca_varcum = np.cumsum(np.round(SB_pca.explained_variance_ratio_, decimals=4)*100)


# In[15]:


plt.grid()
plt.plot(SB_pca_varcum)


# In[55]:


# SB_scaled = SB_train.iloc[:,2:201].values


# In[56]:


# SB_scaled_data = scale(SB_scaled)


# In[57]:


# SB_pca_scaled = PCA()
# SB_pca_scaled.fit(SB_scaled_data)


# In[58]:


# SB_pca_scaled_varcum = np.cumsum(np.round(SB_pca_scaled.explained_variance_ratio_, decimals=4)*100)


# In[59]:


# plt.grid()
# plt.plot(SB_pca_scaled_varcum)


# In[60]:


# plt.plot(SB_pca_scaled.explained_variance_ratio_)


# In[61]:


# plt.plot(SB_pca.explained_variance_ratio_)


# In[16]:


SB_pca_sel = PCA(n_components=150)
SB_pca_sel.fit(SB_train.iloc[:,2:201])


# In[17]:


SB_pca_fin = SB_pca_sel.fit_transform(SB_train.iloc[:,2:201])


# In[18]:


print SB_pca_fin


# In[19]:


SB_pca_fin.shape


# In[20]:


def set_colnames(n):
    b = []
    for i in range(1,n+1):
        a = "".join(['pca_',str(i)])
        b.append(a)
    return b

col_names = set_colnames(150)


# In[21]:


SB_pca_fin_train = pd.DataFrame(SB_pca_fin, columns=col_names)
SB_pca_fin_train.head(10)


# In[22]:


rm.seed(12)
from sklearn.cross_validation import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(SB_pca_fin_train, SB_train['target'], test_size = 0.2, stratify = SB_train['target'])


# In[72]:


y_train.dtypes


# In[24]:


d = {'target' : y_test}


# In[25]:


test_y = pd.DataFrame(data = d)
test_y.head(10)


# In[26]:


from statsmodels.api import Logit


# In[35]:


rm.seed(123)
SB_logit = Logit(y_train, x_train).fit()
SB_logit.summary()


# In[36]:


SB_pred = SB_logit.predict(x_test)


# In[37]:


test_y['predicted'] = SB_pred
test_y.head(10)


# In[38]:


test_y['pred_round'] = 1


# In[39]:


test_y.loc[test_y.predicted < 0.5, 'pred_round'] = 0
test_y.head(10)


# In[40]:


CM = pd.crosstab(test_y['target'], test_y['pred_round'])
CM


# In[47]:


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
FP = CM.iloc[0,1]
TP = CM.iloc[1,1]

print (TP+TN)*100/(TP+FP+TN+FN), TP*100/(TP+FN), TP*100/(TP+FP), TN*100/(TN+FP)


# In[42]:


# accuracy = 58% | recall = 88% | precision = 18% | specifity = 55%


# In[48]:


# Decision Tree
from sklearn import tree


# In[49]:


rm.seed(124)
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)


# In[50]:


C50_pred = C50_model.predict(x_test)


# In[51]:


CM = pd.crosstab(test_y['target'], C50_pred)
CM


# In[52]:


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
FP = CM.iloc[0,1]
TP = CM.iloc[1,1]

print (TP+TN)*100/(TP+FP+TN+FN), TP*100/(TP+FN), TP*100/(TP+FP), TN*100/(TN+FP)


# In[53]:


# accuracy = 83% | recall = 18% | precision = 17% | Specificity = 90%


# In[54]:


# Random forest
from sklearn.ensemble import RandomForestClassifier


# In[55]:


rm.seed(125)
RF_model = RandomForestClassifier(n_estimators=5).fit(x_train, y_train)


# In[56]:


RF_pred = RF_model.predict(x_test)


# In[57]:


CM = pd.crosstab(test_y['target'], RF_pred)
CM


# In[59]:


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
FP = CM.iloc[0,1]
TP = CM.iloc[1,1]

print (TP+TN)*100/(TP+FP+TN+FN), TP*100/(TP+FN), TP*100/(TP+FP), TN*100/(TN+FP)


# In[99]:


# accuracy = 89% | recall = 6% | precision = 30% | specificity = 98%


# In[60]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB


# In[61]:


rm.seed(126)
NB_model = GaussianNB().fit(x_train, y_train)


# In[62]:


#predict test cases
NB_Pred = NB_model.predict(x_test)


# In[63]:


CM = pd.crosstab(test_y['target'], NB_Pred)
CM


# In[64]:


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
FP = CM.iloc[0,1]
TP = CM.iloc[1,1]

print (TP+TN)*100/(TP+FP+TN+FN), TP*100/(TP+FN), TP*100/(TP+FP), TN*100/(TN+FP)


# In[106]:


# accuracy = 91% | recall = 21% | precision = 73% | specificity = 99%


# In[107]:


# Test data & running Logistic regression on it


# In[65]:


SB_test = pd.read_csv("test.csv")


# In[66]:


SB_test.head(10)


# In[67]:


SB_pca_test = SB_pca_sel.fit_transform(SB_test.iloc[:,1:200])


# In[68]:


SB_pca_fin_test = pd.DataFrame(SB_pca_test, columns=col_names)


# In[69]:


SB_pca_fin_test.head(10)


# In[70]:


predicted = NB_model.predict(SB_pca_fin_test)


# In[71]:


SB_test['target'] = 1
SB_test.loc[predicted < 0.5, 'target'] = 0


# In[72]:


SB_test.to_csv("Test_output.csv")

