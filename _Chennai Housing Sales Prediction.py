#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv("C:/Users/navir/Downloads/Chennai housing.csv")
df1.head()


# In[5]:


df1.groupby('Area')['Area'].agg('count')


# In[6]:


df2 = df1.drop(['Condition','Parking','Date Sale','Rooms'],axis='columns')


# In[7]:


df2


# In[8]:


df2.isnull().sum()


# In[9]:


df3 = df2.dropna()


# In[10]:


df3.isnull().sum()


# In[11]:


df2.shape


# In[12]:


df3.shape


# In[13]:


df3['Area'].unique()


# In[14]:


df3['Bedrooms'].unique()


# In[15]:


df3['Total_Sqft'].unique()


# In[16]:


df4 = df3.copy()
df4['price_per_sqft'] = df4['Sales_Price']/df4['Total_Sqft']
df4.head()


# In[17]:


df4.Area = df4.Area.apply(lambda x: x.strip())


# In[18]:


Area_stats = df4.groupby('Area')['Area'].agg('count').sort_values(ascending=False)
Area_stats


# In[19]:


len(Area_stats[Area_stats<=10])


# In[20]:


Area_stats_less_than_10 = Area_stats[Area_stats<=10]
Area_stats_less_than_10


# In[21]:


len(df4.Area.unique())


# In[22]:


df4.Area = df4.Area.apply(lambda x: 'other' if x in Area_stats_less_than_10 else x)
len(df4.Area.unique())


# In[23]:


df4.head()


# In[24]:


df4['Bedrooms'] = df4['Bedrooms'].astype(int)
df4


# In[25]:


df4[df4.Total_Sqft/df4.Bedrooms<300].head()


# In[26]:


df6 = df4[~df4.Total_Sqft/df4.Bedrooms<300]
df6.shape


# In[27]:


df6.price_per_sqft.describe()


# In[28]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Area'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[34]:


def plot_scatter_chart(df,Area):
    bhk2 = df[(df.Area==Area)&(df.Bedrooms==2)]
    bhk3 = df[(df.Area==Area)&(df.Bedrooms==3)]
    matplotlib.rcParams['figure.figsize']=(15,20)
    plt.scatter(bhk2.Total_Sqft,bhk2.price_per_sqft, color='blue',label='2BHK',s=50)
    plt.scatter(bhk3.Total_Sqft,bhk3.price_per_sqft,marker='+',color='green',label='3BHK',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(Area)
    plt.legend()

plot_scatter_chart(df7,"Velachery")
    
    
    


# In[42]:


def remove_bhk_outliers (df):
    exclude_indices = np.array([])
    for Area, Area_df in df.groupby('Area'):
        Bedrooms_stats = {}
        for Bedrooms,Bedrooms_df in Area_df.groupby('Bedrooms'):
            Bedrooms_stats [Bedrooms] = {
                'mean': np.mean (Bedrooms_df.price_per_sqft), 
                'std': np.std (Bedrooms_df.price_per_sqft),
                'count': Bedrooms_df.shape[0]
            }
        for  Bedrooms, Bedrooms_df in Area_df.groupby('Bedrooms'):
            stats =  Bedrooms_stats.get(Bedrooms-1)
            if stats and stats['count']>5:
                    exclude_indices = np.append(exclude_indices, Bedrooms_df [ Bedrooms_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)
df8.shape


# In[56]:


df8[df8.Bathrooms>df8.Bedrooms+2]
df9 = df8[df8.Bathrooms<df8.Bedrooms+2]
df9


# In[58]:


df10 = df9.drop(['Bedrooms','price_per_sqft'],axis='columns')
df10.head(3)


# In[62]:


dummies = pd.get_dummies(df10.Area)
dummies


# In[64]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head(3)


# In[66]:


df12 = df11.drop('Area',axis='columns')
df12.head(2)


# In[68]:


X = df12.drop('Sales_Price',axis='columns')
X.head(2)


# In[69]:


y = df12.Sales_Price
y.head()


# In[71]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[72]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[77]:


from sklearn.model_selection import ShuffleSplit 
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)


# In[87]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso 
from sklearn.tree import DecisionTreeRegressor
def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(), 
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(), 
            'params': {
                'criterion': ['mse', 'friedman_mse'], 
                'splitter': ['best', 'random']
            }
        }

    } 

    scores = []
    cv = ShuffleSplit (n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
find_best_model_using_gridsearchcv(X,y)


# In[89]:


def predict_price (Area, sqft, Bathrooms, Bedrooms): 
    loc_index = np.where(X.columns==Area)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = Bathrooms
    x[2] = Bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]


# In[91]:


predict_price('Adyar',1000,2,2)


# In[92]:


predict_price('Velachery',1000,2,2)


# In[ ]:




