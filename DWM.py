#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
from chart_studio.plotly import iplot
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("dataDWM.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.nunique()


# In[6]:


df.isna().any()


# In[7]:


df=df.drop(columns=["ID","Z_CostContact", "Z_Revenue"],axis=1)
df.head()


# In[8]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[9]:


df['Income'] = df['Income'].fillna(df['Income'].mean())
df.isna().any()


# In[10]:


df['Marital_Status'].value_counts()


# In[11]:


df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')


# In[12]:


df['Marital_Status'].value_counts()  


# In[13]:


df['Education'].value_counts()


# In[14]:


df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'higher')  


# In[15]:


df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']
df['Age'] = 2021 - df["Year_Birth"]


# In[16]:


df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['first_day'] = '15-10-2021'
df['first_day'] = pd.to_datetime(df.first_day)
df['day_engaged'] = (df['first_day'] - df['Dt_Customer']).dt.days


# In[17]:


col_del = ["Dt_Customer", "first_day", "Year_Birth", "Dt_Customer", "Recency", "Complain","AcceptedCmp1" , "AcceptedCmp2",
           "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5", "Response","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases",
           "NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", 
           "MntSweetProducts", "MntGoldProds"]
df=df.drop(columns=col_del,axis=1)
df.head()


# In[18]:


df.shape


# In[29]:


fig = px.bar(df, x='Marital_Status', y='Expenses', color="Education" )
fig.show()


# In[23]:


fig = px.bar(df, x='Marital_Status', y='Expenses', color="Marital_Status")
fig.show()


# In[20]:


fig = px.histogram (df, x = "Expenses",  facet_row = "Marital_Status")
fig.show ()


# In[21]:


fig = px.histogram (df, x = "Expenses",  facet_row = "Education")
fig.show ()


# In[22]:


fig = px.histogram (df, x = "NumTotalPurchases",  facet_row = "Education")
fig.show ()


# In[23]:


fig = px.histogram (df, x = "Age",  facet_row = "Marital_Status")
fig.show ()


# In[24]:


fig = px.histogram (df, x = "Income",  facet_row = "Marital_Status")
fig.show ()


# In[25]:


fig =  px.pie (df, names = "Marital_Status", hole = 0.4)
fig.show ()


# In[26]:


fig =  px.pie (df, names = "Education", hole = 0.4)
fig.show ()


# In[27]:


sns.barplot(x = df['Expenses'],y = df['Education']);
plt.title('Total Expense based on the Education Level');


# In[28]:


sns.barplot(x = df['Income'],y = df['Education']);
plt.title('Total Income based on the Education Level');


# In[29]:


df.describe()


# In[30]:


sns.heatmap(df.corr(), annot=True)


# In[31]:


df.info()


# In[32]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
r = []
for i in df.columns:
    if (df[i].dtypes == "object"):
        r.append(i)
lbl_encode = LabelEncoder()
for i in r:
    df[i]=df[[i]].apply(lbl_encode.fit_transform)


# In[33]:


df['Marital_Status'].value_counts()


# In[34]:


df['Education'].value_counts()


# In[35]:


df1 = df.copy()
scaled_features = StandardScaler().fit_transform(df1.values)
X = pd.DataFrame(scaled_features, index=df1.index, columns=df1.columns)
X.head()


# In[36]:


from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
 kmeans=KMeans(n_clusters=i,init='k-means++',random_state=100)
 kmeans.fit(X)
 wcss.append(kmeans.inertia_)
plt.figure(figsize=(16,8))
plt.plot(range(1,11),wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[37]:


from sklearn.metrics import silhouette_score 
silhouette_scores = []
for i in range(2,10):
    m1=KMeans(n_clusters=i, random_state=100)
    c = m1.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, m1.fit_predict(X))) 
silhouette_scores


# In[38]:


print("number of cluster with max silhoutte score =" , (silhouette_scores.index(max(silhouette_scores))+2))


# In[39]:


kmeans=KMeans(n_clusters=3, random_state=100)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
df['cluster'] =  y_kmeans + 1


# In[40]:


df.head()


# In[41]:


df['cluster'].value_counts()


# In[42]:


pl = sns.countplot(x=df["cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()


# In[43]:


j=0
for i in df:
    if j==9: break;
    diag = sns.FacetGrid(df, col = "cluster", hue = "cluster", palette = "Set1")
    diag.map(plt.hist, i, bins=6, ec="k")
    j+=1


# In[44]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)


# In[45]:


test_y=test['Expenses']


# In[46]:


test_x=test.drop(columns=['Expenses','cluster'],axis=1)


# In[47]:


train_x=train.drop(columns=['Expenses','cluster'],axis=1)


# In[48]:


train_y=train['Expenses']


# In[49]:


from sklearn import linear_model, metrics
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)


# In[50]:


print('Coefficients: ', reg.coef_)


# In[51]:


from sklearn.metrics import mean_squared_error
train_pred=reg.predict(train_x)
test_pred=reg.predict(test_x)
print("training MSE =" ,mean_squared_error(train_y,train_pred))
print("test MSE =" ,mean_squared_error(test_y,test_pred))


# In[52]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500)
regressor.fit(train_x, train_y)
train_pred=regressor.predict(train_x)
test_pred=regressor.predict(test_x)
print("training MSE =" ,mean_squared_error(train_y,train_pred))
print("test MSE =" ,mean_squared_error(test_y,test_pred))


# In[ ]:




